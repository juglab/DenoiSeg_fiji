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

import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.display.Display;
import org.scijava.display.DisplayService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

public class PreviewHandler {

	@Parameter
	private UIService uiService;

	private final int trainDimensions;
	private RandomAccessibleInterval<FloatType> trainingImage;
	private RandomAccessibleInterval<FloatType> validationImage;

	private RandomAccessibleInterval<FloatType> inputImage;
	private RandomAccessibleInterval<FloatType> outputDenoiseImage;
	private RandomAccessibleInterval<FloatType> outputSegmentImage;

	public PreviewHandler(Context context, int trainDimensions) {
		context.inject(this);
		this.trainDimensions = trainDimensions;
	}

	public void updateValidationPreview(RandomAccessibleInterval<FloatType> in,
	                                    RandomAccessibleInterval<FloatType> outDenoise,
	                                    RandomAccessibleInterval<FloatType> outSegment,
	                                    boolean isHeadless, DenoiSegOutputHandler outputHandler, boolean canceledOrStopped) {

		if (Thread.interrupted()) return;

		setSamples(denormalize(in, outputHandler), denormalize(outDenoise, outputHandler), outSegment);

		if(isHeadless || canceledOrStopped) return;

		long[] dims = new long[in.numDimensions()-1];
		int channelCount = 5;
		dims[0] = in.dimension(0)* channelCount;
		for (int i = 1; i < dims.length; i++) {
			dims[i] = in.dimension(i);
		}
		if(validationImage == null) {
			validationImage = new ArrayImgFactory<>(new FloatType()).create(dims);
		}
		long[] minSize = new long[dims.length*2];
		for (int i = 0; i < dims.length; i++) {
			minSize[i] = 0;
			minSize[i+dims.length] = in.dimension(i);
		}
		for (int i = 0; i < channelCount; i++) {
			minSize[0] = i*in.dimension(0);
			FinalInterval interval = Intervals.createMinSize(minSize);
			RandomAccessibleInterval<FloatType> source;
			if(i == 0) {
				source = Views.hyperSlice(in, dims.length, 0);
			} else if(i < 2) {
				source = Views.hyperSlice(outDenoise, dims.length, i-1);
			}
			else {
				source = Views.hyperSlice(outSegment, dims.length, i-2);
			}
			LoopBuilder.setImages(Views.zeroMin(Views.interval(validationImage, interval)), source)
					.multiThreaded().forEachPixel(FloatType::set);
		}

//		else LoopBuilder.setImages(splitImage, out).forEachPixel((o, i) -> o.set(i));
//		else opService.copy().rai(splitImage, (RandomAccessibleInterval)out);
//		if(trainDimensions == 2) updateSplitImage2D(in);
//		if(trainDimensions == 3) updateSplitImage3D(in);
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training preview");
		if(display == null) uiService.show("training preview", validationImage);
		else display.update();
	}

	private RandomAccessibleInterval<FloatType> denormalize(RandomAccessibleInterval<FloatType> img, DenoiSegOutputHandler outputHandler) {
		return de.csbdresden.n2v.train.TrainUtils.denormalizeConverter(img, outputHandler.getMean(), outputHandler.getStdDev());
	}

	private void setSamples(RandomAccessibleInterval<FloatType> in, RandomAccessibleInterval<FloatType> outDenoise, RandomAccessibleInterval<FloatType> outSegment) {
		inputImage = Views.hyperSlice(in, in.numDimensions()-2, 0);
		outputDenoiseImage = outDenoise;
		outputSegmentImage = outSegment;
	}

	RandomAccessibleInterval<FloatType> getExampleInput() {
		return inputImage;
	}

	RandomAccessibleInterval<FloatType> getExampleOutputDenoise() {
		return outputDenoiseImage;
	}

	RandomAccessibleInterval<FloatType> getExampleOutputSegment() {
		return outputSegmentImage;
	}
}
