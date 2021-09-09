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

import io.bioimage.specification.ModelSpecification;
import net.imagej.modelzoo.ImageTensorSample;
import net.imagej.modelzoo.consumer.model.tensorflow.train.AbstractOutputHandler;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.Context;
import org.tensorflow.Graph;
import org.tensorflow.Session;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class DenoiSegOutputHandler extends AbstractOutputHandler {

	private final DenoiSegConfig config;
	private final DenoiSegTraining training;
	private FloatType mean = new FloatType();

	private FloatType stdDev = new FloatType();

	private float currentLearningRate;
	private float currentLoss = Float.MAX_VALUE;
	private float currentDenoiseLoss = Float.MAX_VALUE;
	private float currentSegLoss = Float.MAX_VALUE;
	private float currentValidationLoss = Float.MAX_VALUE;
	private float bestValidationLoss = Float.MAX_VALUE;

	private final static String defaultSampleInput = "sample_in.tif";
	private final static String defaultSampleOutput = "sample_out.tif";
	private double threshold = 0.5; //TODO calculate from validation data

	DenoiSegOutputHandler(DenoiSegConfig config, DenoiSegTraining training, Context context) {
		super(context);
		this.config = config;
		this.currentLearningRate = config.getLearningRate();
		this.training = training;
	}

	@Override
	protected String getShortName() {
		return "denoiseg";
	}

	@Override
	protected ModelSpecification createSpecification(String name) {
		DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
		spec.setName("unnamed denoiseg " + name);
		spec.setTimestamp(getTimestamp());
		spec.update(config, this, training.getStepsFinished());
		return spec;
	}

	void createSavedModelDirs() throws IOException {
		String predictionGraphDir = config.getTrainDimensions() == 2 ? "denoiseg_prediction_2d" : "denoiseg_prediction_3d";
		super.createSavedModelDirs(getClass().getResourceAsStream("/" + predictionGraphDir + "/saved_model.pb"));
	}

	void loadUntrainedGraph(Graph graph) throws IOException {
		String graphName = config.getTrainDimensions() == 2 ? "denoiseg_graph_2d.pb" : "denoiseg_graph_3d.pb";
		super.loadUntrainedGraph(graph, getClass().getResourceAsStream("/" + graphName) );
	}

	@Override
	protected void copyBestModel() {
		if(bestValidationLoss > currentValidationLoss) {
			bestValidationLoss = currentValidationLoss;
			super.copyBestModel();
		}
	}

	void saveCheckpoint(Session sess, RandomAccessibleInterval<FloatType> exampleInput,
	                    RandomAccessibleInterval<FloatType> exampleOutput) {
		super.saveCheckpoint(sess,
				Collections.singletonList(new ImageTensorSample<>(exampleInput, defaultSampleInput)),
				Collections.singletonList(new ImageTensorSample<>(exampleOutput, defaultSampleOutput)));
	}

	float getCurrentSegLoss() {
		return currentSegLoss;
	}

	float getCurrentDenoiseLoss() {
		return currentDenoiseLoss;
	}

	float getCurrentLearningRate() {
		return currentLearningRate;
	}

	void setCurrentLearningRate(float rate) {
		currentLearningRate = rate;
	}

	public FloatType getMean() {
		return mean;
	}

	public FloatType getStdDev() {
		return stdDev;
	}

	void setCurrentDenoiseLoss(float loss) {
		this.currentDenoiseLoss = loss;
	}

	void setCurrentSegLoss(float abs) {
		this.currentSegLoss = abs;
	}

	float getCurrentValidationLoss() {
		return currentValidationLoss;
	}

	void setCurrentValidationLoss(float loss) {
		this.currentValidationLoss = loss;
	}

	void setCurrentLoss(float loss) {
		this.currentLoss = loss;
	}

	float getCurrentLoss() {
		return currentLoss;
	}

	List<String> getSampleInputNames() {
		return Collections.singletonList(defaultSampleInput);
	}

	List<String> getSampleOutputNames() {
		return Collections.singletonList(defaultSampleOutput);
	}
}
