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
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.consumer.AbstractModelZooPrediction;
import net.imagej.modelzoo.consumer.ModelZooPredictionOptions;
import net.imagej.modelzoo.consumer.SingleImagePrediction;
import net.imagej.modelzoo.consumer.model.ModelZooModel;
import net.imagej.modelzoo.consumer.model.node.ImageDataReference;
import net.imagej.modelzoo.consumer.model.prediction.ImageInput;
import net.imagej.modelzoo.consumer.sanitycheck.SanityCheck;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import org.scijava.Context;
import org.scijava.plugin.Parameter;

import java.io.File;

public class DenoiSegPrediction extends AbstractModelZooPrediction<ImageInput<?>, DenoiSegOutput<?, ?>> implements SingleImagePrediction<DenoiSegOutput<?, ?>> {

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	public DenoiSegPrediction() {
	}

	public DenoiSegPrediction(Context context) {
		super(context);
	}

	@Override
	protected DenoiSegOutput<?, ?> createOutput(ModelZooModel model) {
		ImageDataReference<?> denoised = (ImageDataReference<?>) model.getOutputNodes().get(0).getData();
		ImageDataReference<?> segmented = (ImageDataReference<?>) model.getOutputNodes().get(1).getData();
		return new DenoiSegOutput<>(denoised.getData(), segmented.getData());
	}

	@Override
	public SanityCheck getSanityCheck() {
		// there is no sanity check implemented for segmentation models yet
		return null;
	}

	@Override
	public boolean canRunSanityCheck(ModelZooArchive trainedModel) {
		return false;
	}

	public <T extends RealType<T> & NativeType<T>> DenoiSegOutput<?, ?> predict(RandomAccessibleInterval<T> input, String axes) throws Exception {
		String inputName = getTrainedModel().getSpecification().getInputs().get(0).getName();
		setInput(new ImageInput<>(inputName, input, axes));
		run();
		return getOutput();
	}
}
