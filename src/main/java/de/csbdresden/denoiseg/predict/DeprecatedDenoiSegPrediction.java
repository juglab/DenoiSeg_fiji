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

import net.imagej.ImageJ;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.modelzoo.consumer.ModelZooPredictionOptions;
import net.imagej.modelzoo.consumer.model.ModelZooModel;
import net.imagej.modelzoo.consumer.model.node.ImageDataReference;
import net.imagej.modelzoo.consumer.sanitycheck.SanityCheck;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;

@Plugin(type = ModelZooPrediction.class, name = "denoiseg")
public class DeprecatedDenoiSegPrediction extends DenoiSegPrediction {

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	public DeprecatedDenoiSegPrediction(Context context) {
		super(context);
	}

	@Override
	protected DenoiSegOutput<?, ?> createOutput(ModelZooModel model) {
		ImageDataReference<?> data = (ImageDataReference<?>) model.getOutputNodes().get(0).getData();
		IntervalView denoised = getFirstChannel(data.getData());
		IntervalView segmented = getSegmentationChannels(data.getData());
		clip(segmented);
		return new DenoiSegOutput<>(denoised, segmented);
	}

	private <T extends RealType<T>> void clip(RandomAccessibleInterval<T> img) {
		LoopBuilder.setImages(img).forEachPixel(t -> t.setReal(Math.max(0, Math.min(t.getRealDouble(), 1))));
	}

	private <T> IntervalView<T> getFirstChannel(RandomAccessibleInterval<T> output) {
		long[] dims = new long[output.numDimensions()];
		output.dimensions(dims);
		dims[dims.length-1] = 1;
		return Views.interval(output, new FinalInterval(dims));
	}

	private <T> IntervalView<T> getSegmentationChannels(RandomAccessibleInterval<T> output) {
		long[] minmax = new long[output.numDimensions()*2];
		for (int i = 0; i < output.numDimensions()-1; i++) {
			minmax[i] = 0;
			minmax[i+output.numDimensions()] = output.dimension(i);
		}
		minmax[output.numDimensions()-1] = 1;
		minmax[output.numDimensions()*2-1] = 3;
		return Views.interval(output, Intervals.createMinSize(minmax));
	}

	@Override
	public SanityCheck getSanityCheck() {
		// there is no sanity check implemented for segmentation models yet
		return null;
	}

	public static void main(final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );
		String modelFile = "/home/random/Development/imagej/project/CSBDeep/training/DenoiSeg/mouse/latest.modelzoo.zip";
		final File predictionInput = new File( "/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train/img_3.tif" );

		RandomAccessibleInterval _input = ( RandomAccessibleInterval ) ij.io().open( predictionInput.getAbsolutePath() );
		RandomAccessibleInterval _inputConverted = ij.op().copy().rai(ij.op().convert().float32( Views.iterable( _input )));

		DeprecatedDenoiSegPrediction prediction = new DeprecatedDenoiSegPrediction(ij.context());
		prediction.setTrainedModel(modelFile);
		prediction.setOptions(ModelZooPredictionOptions.options().numberOfTiles(1));
		DenoiSegOutput output = prediction.predict(_inputConverted, "XY");
		ij.ui().show( "denoised", output.getDenoised() );
		ij.ui().show( "segmented", output.getSegmented() );

	}
}
