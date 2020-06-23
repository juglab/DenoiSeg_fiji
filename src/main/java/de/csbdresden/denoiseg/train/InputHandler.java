/*-
 * #%L
 * DenoiSeg plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package de.csbdresden.denoiseg.train;

import de.csbdresden.n2v.ui.TrainingProgress;
import net.imagej.modelzoo.consumer.converter.RealIntConverter;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.boundary.IntTypeBoundary;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.event.EventService;
import org.scijava.io.IOService;
import org.scijava.io.event.IOEvent;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class InputHandler {

	@Parameter
	private LogService logService;

	@Parameter
	private IOService ioService;

	@Parameter
	private EventService eventService;

	@Parameter
	private OpService opService;

	private final DenoiSegConfig config;
	private TrainingProgress dialog;

	private int showXPreviews = 10;

	private final List< RandomAccessibleInterval< FloatType > > trainingUnlabeled = new ArrayList<>();
	private final XYPairs<FloatType> trainingLabeled = new XYPairs<>();
	private final XYPairs<FloatType> validationData = new XYPairs<>();
	private final XYPairs<FloatType> trainingData = new XYPairs<>();
	private Collection subscribers;
	private boolean canceled = false;

	InputHandler(Context context, DenoiSegConfig config) {
		this.config = config;
		context.inject(this);
	}

	void setDialog(TrainingProgress dialog) {
		this.dialog = dialog;
	}

	private void unregisterIOEvent() {
		subscribers = eventService.getSubscribers(IOEvent.class);
		eventService.unsubscribe(subscribers);
	}

	private void registerIOEvent() {
		eventService.subscribe(subscribers);
	}

	private RandomAccessibleInterval<FloatType> convertToOneHot(RandomAccessibleInterval<IntType> labeling) {
		RandomAccessibleInterval<FloatType> background = new ArrayImgFactory<>(new FloatType()).create(labeling);
		RandomAccessibleInterval<FloatType> foreground = new ArrayImgFactory<>(new FloatType()).create(labeling);
		RandomAccessibleInterval<FloatType> border = new ArrayImgFactory<>(new FloatType()).create(labeling);
		IntTypeBoundary<IntType> boundary = new IntTypeBoundary<>(labeling);
		LoopBuilder.setImages(labeling, boundary, border, background, foreground).forEachPixel((in, bound, bord, back, fore) -> {
			if (bound.get() != 0) bord.setOne();
			else {
				if (in.get() == 0) back.setOne();
				else fore.setOne();
			}
		});
		return Views.stack(background, foreground, border);
	}

	private IntervalView<FloatType> addTwoDimensions(RandomAccessibleInterval<FloatType> channel0) {
		return Views.addDimension(Views.addDimension(channel0, 0, 0), 0, 0);
	}

	private <T extends RealType<T>> RandomAccessibleInterval<FloatType> convertToFloat(RandomAccessibleInterval<T> img) {
		return opService.copy().rai( (RandomAccessibleInterval<FloatType>)Views.iterable(Converters.convert(img, new RealFloatConverter<T>(), new FloatType())));
	}

	public void addTrainingData(File trainingRawData, File trainingLabelingData) throws IOException {

		logService.info( "Tile training data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling training data" );

		unregisterIOEvent();

		for (File file : trainingRawData.listFiles()) {
			if(canceled) break;
			if(file.isDirectory()) continue;
//					System.out.println(file.getAbsolutePath());
			Img image = (Img) ioService.open(file.getAbsolutePath());
			if(image == null) continue;
			RandomAccessibleInterval<IntType> labeling = getLabeling(file, trainingLabelingData);
			RandomAccessibleInterval<FloatType> imageFloat = convertToFloat(image);
			addTrainingData(imageFloat, labeling);
		}

		registerIOEvent();
	}

	private RandomAccessibleInterval<IntType> getLabeling(File rawFile, File labelingDirectory) {
		for (File labeling : labelingDirectory.listFiles()) {
			if(canceled) break;
			if(rawFile.getName().equals(labeling.getName())) {
				try {
					RandomAccessibleInterval label = (Img) ioService.open(labeling.getAbsolutePath());
					return convertToInt(label);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return null;
	}

	private static <T extends RealType<T>> RandomAccessibleInterval<IntType> convertToInt(RandomAccessibleInterval<T> img) {
		return Converters.convert(img, new RealIntConverter<T>(), new IntType());
	}

	public void addTrainingData(RandomAccessibleInterval<FloatType> raw, RandomAccessibleInterval<IntType> labeling) {

		if (Thread.interrupted()) return;

//		logService.info("Training image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(raw)));

		if(labeling != null) {
			RandomAccessibleInterval<FloatType> oneHot = convertToOneHot(labeling);
			List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> tiles = DenoiSegDataGenerator.createTiles(raw, oneHot, config.getTrainDimensions(), config.getTrainPatchShape(), logService);
			for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> tile : tiles) {
				RandomAccessibleInterval<FloatType> channel0 = addTwoDimensions(tile.getA());
				RandomAccessibleInterval<FloatType> channel1 = addBatchDimension(tile.getB());
				trainingLabeled.add(new ValuePair<>(channel0, channel1));
			}
		} else {
			List<RandomAccessibleInterval<FloatType>> tiles = DenoiSegDataGenerator.createTiles(raw, config.getTrainDimensions(), config.getTrainPatchShape(), logService);
			for (RandomAccessibleInterval<FloatType> tile : tiles) {
				trainingUnlabeled.add(addTwoDimensions(tile));
			}
		}
	}

	public void addValidationData(File validationRawData, File validationLabelingData) throws IOException {

		logService.info( "Tile validation data.." );
		if(dialog != null) dialog.setCurrentTaskMessage("Tiling validation data" );

		unregisterIOEvent();

		for (File file : validationRawData.listFiles()) {
			if(canceled) break;
			if(file.isDirectory()) continue;
			Img image = (Img) ioService.open(file.getAbsolutePath());
			RandomAccessibleInterval<IntType> labeling = getLabeling(file, validationLabelingData);
			RandomAccessibleInterval<FloatType> imageFloat = convertToFloat(image);
			addValidationData(imageFloat, labeling);
		}

		registerIOEvent();

	}

	public void addValidationData(RandomAccessibleInterval<FloatType> validationRaw, RandomAccessibleInterval<IntType> validationLabeling) {
		if(validationLabeling == null) {
			logService.warn("Validation data without labeling is ignored (this will be improved in the future)");
			return;
		}

		if (Thread.interrupted()) return;

		//		logService.info("Validation image dimensions: " + Arrays.toString(Intervals.dimensionsAsIntArray(validation)));

		RandomAccessibleInterval<FloatType> oneHot = convertToOneHot(validationLabeling);

		List<Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>> tiles =
				DenoiSegDataGenerator.createTiles(validationRaw, oneHot, config.getTrainDimensions(), config.getTrainPatchShape(), logService);
		for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> pair : tiles) {
			RandomAccessibleInterval<FloatType> channel0 = addTwoDimensions(pair.getA());
			RandomAccessibleInterval<FloatType> channel1 = addBatchDimension(pair.getB());
			validationData.add(new ValuePair<>(channel0, channel1));
		}
	}

	private RandomAccessibleInterval<FloatType> addBatchDimension(RandomAccessibleInterval<FloatType> img) {
		img = Views.addDimension(img, 0, 0);
		return Views.moveAxis(img, img.numDimensions()-1, img.numDimensions()-2);
	}

	void finalizeTrainingData() {
//		Collections.shuffle(trainingUnlabeled);
//		Collections.shuffle(trainingLabeled);
		Collections.shuffle(validationData);
		trainingData.clear();
		trainingData.addAll(trainingLabeled);
		ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		for (RandomAccessibleInterval<FloatType> raw : trainingUnlabeled) {
			long[] dims = new long[raw.numDimensions()];
			raw.dimensions(dims);
			dims[dims.length-1] = 3;
			trainingData.add(new ValuePair<>(raw, factory.create(dims)));
		}
		Collections.shuffle(trainingData);
	}

	XYPairs<FloatType> getTrainingData() {
		return trainingData;
	}

	XYPairs<FloatType> getValidationData() {
		return validationData;
	}

	public XYPairs<FloatType> getLabeledTrainingPairs() {
		return trainingLabeled;
	}

	public void cancel() {
		this.canceled = true;
	}
}
