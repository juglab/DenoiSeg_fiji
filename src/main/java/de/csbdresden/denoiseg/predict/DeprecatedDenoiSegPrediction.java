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

import de.csbdresden.n2v.util.N2VUtils;
import io.scif.img.converters.RandomAccessConverter;
import net.imagej.modelzoo.consumer.ModelZooPrediction;
import net.imagej.modelzoo.consumer.model.ModelZooModel;
import net.imagej.modelzoo.consumer.model.node.ImageDataReference;
import net.imagej.modelzoo.consumer.model.node.InputImageNode;
import net.imagej.modelzoo.consumer.model.node.ModelZooNode;
import net.imagej.modelzoo.consumer.model.node.OutputImageNode;
import net.imagej.modelzoo.consumer.sanitycheck.SanityCheck;
import net.imagej.modelzoo.plugin.transformation.postprocessing.ScaleLinearPostprocessing;
import net.imagej.modelzoo.plugin.transformation.preprocessing.ZeroMeanUnitVariancePreprocessing;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.plugin.Plugin;

@Plugin(type = ModelZooPrediction.class, name = "denoiseg")
public class DeprecatedDenoiSegPrediction extends DenoiSegPrediction {

	public DeprecatedDenoiSegPrediction() {
		super();
	}
	public DeprecatedDenoiSegPrediction(Context context) {
		super(context);
	}

	@Override
	protected DenoiSegOutput<?, ?> createOutput(ModelZooModel model) {
		ModelZooNode<?> modelZooNode = model.getOutputNodes().get(0);
		ImageDataReference<?> data = (ImageDataReference<?>) modelZooNode.getData();
		RandomAccessibleInterval denoised = getFirstChannel(data.getData());
		ZeroMeanUnitVariancePreprocessing preprocessor = (ZeroMeanUnitVariancePreprocessing) model.getInputNodes().get(0).getProcessors().get(0);
		denoised = denormalize(modelZooNode, denoised, data, preprocessor.getStdDev().floatValue(), preprocessor.getMean().floatValue());
		IntervalView segmented = getSegmentationChannels(data.getData());
		clip(segmented);
		return new DenoiSegOutput<>(denoised, segmented);
	}

	private <O extends RealType<O> & NativeType<O>> RandomAccessibleInterval denormalize(ModelZooNode<?> modelZooNode, RandomAccessibleInterval<O> in, ImageDataReference<O> outType, float gain, float offset) {
		InputImageNode inputReference = ((OutputImageNode) modelZooNode).getReference();
		O resOutType = outType.getDataType();
		if(inputReference != null && getOptions().values.convertIntoInputFormat()) {
			resOutType = inputReference.getOriginalDataType();
		}
		RandomAccessibleInterval<O> out;
		if(sameType(outType.getDataType(), resOutType)) {
			out = outType.getData();
		} else {
			out = opService.create().img(in, resOutType);
		}
		O finalResOutType = resOutType;
		LoopBuilder.setImages(in, out).forEachPixel((i, o) -> {
			double real = i.getRealDouble() * gain + offset;
			o.setReal(inBounds(real, finalResOutType));
		});
		return out;
	}

	private <T extends RealType<T> & NativeType<T>> double inBounds(double value, T resOutType) {
		return Math.min(Math.max(resOutType.getMinValue(), value), resOutType.getMaxValue());
	}


	protected <I extends RealType<I> & NativeType<I>, O extends RealType<O> & NativeType<O>> boolean sameType(I inType, O outType) {
		return inType.getClass().equals(outType);
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
}
