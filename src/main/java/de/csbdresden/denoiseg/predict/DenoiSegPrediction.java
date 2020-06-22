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
package de.csbdresden.denoiseg.predict;

import de.csbdresden.denoiseg.train.DenoiSegModelSpecification;
import de.csbdresden.denoiseg.train.TrainUtils;
import io.scif.MissingLibraryException;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.DefaultSingleImagePrediction;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.modelzoo.consumer.model.InputImageNode;
import net.imagej.modelzoo.consumer.model.ModelZooAxis;
import net.imagej.modelzoo.consumer.model.ModelZooModel;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Plugin(type = ModelZooPrediction.class, name = "denoiseg")
public class DenoiSegPrediction extends DefaultSingleImagePrediction<FloatType, FloatType> {

	private FloatType mean;
	private FloatType stdDev;

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

	@Override
	public void setInput(String name, RandomAccessibleInterval<?> value, String axes) {
		preprocessInput(value, mean, stdDev);
		super.setInput(name, value, axes);
	}

	@Override
	public void run() throws OutOfMemoryError, FileNotFoundException, MissingLibraryException {
		//		super.run();
		ModelZooModel model = loadModel(getTrainedModel());
		if (model != null && model.isInitialized() && this.inputValidationAndMapping(model)) {
			increaseHalo(model);
			try {
				this.preprocessing(model);
				this.executePrediction(model);
				this.postprocessing(model);
			} finally {
				model.dispose();
			}

		} else {
			context.service(LogService.class).error("Model does not exist or cannot be loaded. Exiting.");
			if (model != null) {
				model.dispose();
			}

		}
		postprocessOutput(getOutput(), mean, stdDev);

	}

	private void increaseHalo(ModelZooModel model) {
		//TODO HACK to make tiling work. without increasing the halo the tiles become visible. something's calculated wrong at the border.
		InputImageNode<?> inputNode = model.getInputNodes().get(0);
		for (ModelZooAxis axis : inputNode.getAxes()) {
			if(axis.getHalo() > 1) axis.setHalo(axis.getHalo()+32);
		}
	}

	private void preprocessInput(RandomAccessibleInterval input, FloatType mean, FloatType stdDev) {
		TrainUtils.normalizeInplace(input, mean, stdDev);
	}

	private void postprocessOutput(RandomAccessibleInterval<FloatType> output, FloatType mean, FloatType stdDev) {
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
		setInput(input, axes);
		run();
		if(getOutput() == null) return null;
		return getOutput();
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
