package de.csbdresden.denoiseg.train;

import de.csbdresden.denoiseg.ui.N2VProgress;
import io.scif.services.DatasetIOService;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.roi.boundary.Boundary;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.io.IOService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InputHandler {

	@Parameter
	LogService logService;

	@Parameter
	DatasetIOService datasetIOService;

	@Parameter
	IOService ioService;

	@Parameter
	OpService opService;

	private final DenoiSegConfig config;
	private N2VProgress dialog;

	private final List< RandomAccessibleInterval< FloatType > > X = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > Y = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > validationX = new ArrayList<>();
	private final List< RandomAccessibleInterval< FloatType > > validationY = new ArrayList<>();
	private boolean bigDataMode = false;

	public InputHandler(Context context, DenoiSegConfig config) {
		this.config = config;
		context.inject(this);
	}

	void setDialog(N2VProgress dialog) {
		this.dialog = dialog;
	}

	public void addTrainingAndValidationData(File trainingRawData, File trainingLabelingData) throws IOException {
		for (File file : trainingRawData.listFiles()) {
			Img image = (Img) ioService.open(file.getAbsolutePath());
			Img labeling = getLabeling(file, trainingLabelingData);
			Img imageFloat = opService.convert().float32(Views.iterable(image));
			addTrainingAndValidationData(imageFloat, labeling, 0.1);
		}
	}

	public void addTrainingAndValidationData(RandomAccessibleInterval<FloatType> raw, RandomAccessibleInterval<BitType> labeling, double validationAmount) {
		if (Thread.interrupted()) return;

		logService.info( "Tile training and validation data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling training and validation data" );

		RandomAccessibleInterval<FloatType> oneHot = convertToOneHot(raw, labeling);

		List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> tiles = N2VDataGenerator.createTiles( raw, oneHot, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService );

		int trainEnd = (int) (tiles.size() * (1 - validationAmount));
		for (int i = 0; i < trainEnd; i++) {
			//TODO do I need to copy here?
			X.add(addTwoDimensions(tiles.get(i).getA()));
			Y.add(addBatchDimension(tiles.get( i ).getB()));
		}
		int valEnd = tiles.size()-trainEnd % 2 == 1 ? tiles.size() - 1 : tiles.size();
		for (int i = trainEnd; i < valEnd; i++) {
			//TODO do I need to copy here?
			validationX.add( addTwoDimensions(tiles.get( i ).getA()));
			validationY.add( addBatchDimension(tiles.get( i ).getB()));
		}
	}

	private RandomAccessibleInterval<FloatType> convertToOneHot(RandomAccessibleInterval<FloatType> raw, RandomAccessibleInterval<BitType> labeling) {
		RandomAccessibleInterval<FloatType> background = new DiskCachedCellImgFactory<>(new FloatType()).create(raw);
		RandomAccessibleInterval<FloatType> foreground = new DiskCachedCellImgFactory<>(new FloatType()).create(raw);
		RandomAccessibleInterval<FloatType> border = new DiskCachedCellImgFactory<>(new FloatType()).create(raw);
		if(labeling == null) {
			Views.iterable(background).forEach(FloatType::setOne);
		} else {
			Boundary<BitType> boundary = new Boundary<>(labeling);
			LoopBuilder.setImages(labeling, boundary, border, background, foreground).forEachPixel((in, bound, bord, back, fore) -> {
				if (bound.get()) bord.setOne();
				if (!in.get()) back.setOne();
				if (in.get()) fore.setOne();
			});
		}
		return Views.stack(background, foreground, border);
	}

	private IntervalView<FloatType> addTwoDimensions(RandomAccessibleInterval<FloatType> channel0) {
		return Views.addDimension(Views.addDimension(channel0, 0, 0), 0, 0);
	}

	public static <T extends RealType<T>> RandomAccessibleInterval<FloatType> convertToFloat(RandomAccessibleInterval<T> img) {
		return Converters.convert(img, new RealFloatConverter<T>(), new FloatType());
	}

	public void addTrainingData(File trainingRawData, File trainingLabelingData) throws IOException {

		logService.info( "Tile training data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling training data" );

		for (File file : trainingRawData.listFiles()) {
			if(file.isDirectory()) continue;
//					System.out.println(file.getAbsolutePath());
			Img image = (Img) ioService.open(file.getAbsolutePath());
			if(image == null) continue;
			Img labeling = getLabeling(file, trainingLabelingData);
			Img imageFloat = opService.convert().float32(Views.iterable(image));
			addTrainingData(imageFloat, labeling);
		}
	}

	private Img getLabeling(File rawFile, File labelingDirectory) {
		for (File labeling : labelingDirectory.listFiles()) {
			if(rawFile.getName().equals(labeling.getName())) {
				try {
					return opService.convert().bit((Img)ioService.open(labeling.getAbsolutePath()));
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return null;
	}

	public void addTrainingData(RandomAccessibleInterval<FloatType> raw, RandomAccessibleInterval<BitType> labeling) {

		if (Thread.interrupted()) return;

//		logService.info("Training image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(raw)));

		RandomAccessibleInterval<FloatType> oneHot = convertToOneHot(raw, labeling);

		List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> tiles = N2VDataGenerator.createTiles(raw, oneHot, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService);
		for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> tile : tiles) {

			RandomAccessibleInterval<FloatType> channel0 = addTwoDimensions(tile.getA());
			RandomAccessibleInterval<FloatType> channel1 = addBatchDimension(tile.getB());
			X.add(channel0);
			Y.add(channel1);
		}
	}

	public void addValidationData(File validationRawData, File validationLabelingData) throws IOException {

		logService.info( "Tile validation data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling validation data" );

		for (File file : validationRawData.listFiles()) {
			Img image = (Img) ioService.open(file.getAbsolutePath());
			Img labeling = getLabeling(file, validationLabelingData);
			Img imageFloat = opService.convert().float32(Views.iterable(image));
			addValidationData(imageFloat, labeling);
		}
	}

	public void addValidationData(RandomAccessibleInterval<FloatType> validation, RandomAccessibleInterval<BitType> validationLabeling) {

		if (Thread.interrupted()) return;

		//		logService.info("Validation image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(validation)));

		RandomAccessibleInterval<FloatType> oneHot = convertToOneHot(validation, validationLabeling);

		List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> tiles =
				N2VDataGenerator.createTiles(validation, oneHot, config.getTrainDimensions(), config.getTrainBatchDimLength(), logService);
		for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> tile : tiles) {
			validationX.add(addTwoDimensions(tile.getA()));
			validationY.add(addBatchDimension(tile.getB()));
		}
	}

	private RandomAccessibleInterval<FloatType> addBatchDimension(RandomAccessibleInterval<FloatType> img) {
		Views.addDimension(img, 0, 0);
		return Views.moveAxis(img, img.numDimensions()-1, img.numDimensions()-2);
	}

//	public void addValidationData(File trainingFolder) {
//
//		if(trainingFolder.isDirectory()) {
//			File[] imgs = trainingFolder.listFiles();
//			for (File file : imgs) {
//				if (Thread.interrupted()) return;
//				try {
//					RandomAccessibleInterval img = datasetIOService.open(file.getAbsolutePath()).getImgPlus().getImg();
//					addValidationData(convertToFloat(img));
//				} catch (IOException e) {
//					logService.warn("Could not load " + file.getAbsolutePath() + " as image");
//				}
//			}
//		}
//	}

	public List<RandomAccessibleInterval<FloatType>> getX() {
		return X;
	}
	public List<RandomAccessibleInterval<FloatType>> getY() {
		return Y;
	}

	public List<RandomAccessibleInterval<FloatType>> getValidationX() {
		return validationX;
	}
	public List<RandomAccessibleInterval<FloatType>> getValidationY() {
		return validationY;
	}

	public void setBigDataMode(boolean bigData) {
		this.bigDataMode = bigData;
	}
}
