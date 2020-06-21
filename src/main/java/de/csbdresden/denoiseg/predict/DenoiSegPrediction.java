package de.csbdresden.denoiseg.predict;

import de.csbdresden.denoiseg.train.DenoiSegModelSpecification;
import de.csbdresden.denoiseg.train.TrainUtils;
import io.scif.MissingLibraryException;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.DefaultSingleImagePrediction;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Plugin(type = ModelZooPrediction.class, name = "denoiseg")
public class DenoiSegPrediction extends DefaultSingleImagePrediction<FloatType, FloatType> {

	private FloatType mean;
	private FloatType stdDev;
	private int trainDimensions = 2;

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	public DenoiSegPrediction(Context context) {
		super(context);
	}

	@Override
	public void setTrainedModel(ModelZooArchive trainedModel) {
		super.setTrainedModel(trainedModel);
		DenoiSegModelSpecification.setFromSpecification(this, trainedModel.getSpecification());
	}

	public void setMean(FloatType mean) {
		this.mean = mean;
	}

	public void setStdDev(FloatType stdDev) {
		this.stdDev = stdDev;
	}

	public void setTrainDimensions(int numDimensions) {
		this.trainDimensions = numDimensions;
	}

	@Override
	public void setInput(String name, RandomAccessibleInterval<?> value, String axes) {
		preprocess(value, mean, stdDev);
		super.setInput(name, value, axes);
	}

	@Override
	public void run() throws FileNotFoundException, MissingLibraryException {
		super.run();
		RandomAccessibleInterval output = getOutput();
		postprocess(output, mean, stdDev);
	}

	private void preprocess(RandomAccessibleInterval input, FloatType mean, FloatType stdDev) {
		TrainUtils.normalizeInplace(input, mean, stdDev);
	}

	private void postprocess(RandomAccessibleInterval<FloatType> output, FloatType mean, FloatType stdDev) {
		// only denormalize first channel
		IntervalView<FloatType> firstChannel = getFirstChannel(output);
		TrainUtils.denormalizeInplace(firstChannel, mean, stdDev, opService);
	}

	private IntervalView<FloatType> getFirstChannel(RandomAccessibleInterval<FloatType> output) {
		long[] dims = new long[output.numDimensions()];
		output.dimensions(dims);
		dims[dims.length-1] = 1;
		return Views.interval(output, new FinalInterval(dims));
	}

	public RandomAccessibleInterval<FloatType> predictPadded(RandomAccessibleInterval<FloatType> input, String axes) throws FileNotFoundException, MissingLibraryException {
//		int padding = 32;
//		IntervalView<FloatType> paddedInput = addPadding(input, padding);
		setInput(input, axes);
		run();
		if(getOutput() == null) return null;
		return getOutput();
//		return removePadding(getOutput(), padding);
	}

	private IntervalView<FloatType> addPadding(RandomAccessibleInterval<FloatType> input, int padding) {
		FinalInterval bigger = new FinalInterval(input);
		for (int i = 0; i < trainDimensions; i++) {
			bigger = Intervals.expand(bigger, padding, i);
		}
		return Views.zeroMin(Views.interval(Views.extendMirrorDouble(input), bigger));
	}

	private <T> RandomAccessibleInterval<T> removePadding(RandomAccessibleInterval<T> output, int padding) {
		FinalInterval smaller = new FinalInterval(output);
		for (int i = 0; i < trainDimensions; i++) {
			smaller = Intervals.expand(smaller, -padding, i);
		}
		return Views.zeroMin(Views.interval(output, smaller));
	}

	public static void main( final String... args ) throws IOException, MissingLibraryException {

		final ImageJ ij = new ImageJ();
		ij.launch( args );
		String modelFile = "/home/random/Development/imagej/project/CSBDeep/training/DenoiSeg/mouse/latest.modelzoo.zip";
		final File predictionInput = new File( "/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train/img_3.tif" );

		RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( predictionInput.getAbsolutePath() );
		RandomAccessibleInterval _inputConverted = ij.op().copy().rai(ij.op().convert().float32( Views.iterable( _input )));

		DenoiSegPrediction prediction = new DenoiSegPrediction(ij.context());
		prediction.setTrainedModel(modelFile);
		prediction.setNumberOfTiles(1);
		prediction.setInput(_inputConverted, "XY");
		prediction.run();
		RandomAccessibleInterval output = prediction.getOutput();
		ij.ui().show( output );

	}
}
