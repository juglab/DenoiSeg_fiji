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

import de.csbdresden.denoiseg.predict.DenoiSegPrediction;
import net.imagej.modelzoo.specification.CitationSpecification;
import net.imagej.modelzoo.specification.DefaultCitationSpecification;
import net.imagej.modelzoo.specification.DefaultInputNodeSpecification;
import net.imagej.modelzoo.specification.DefaultModelSpecification;
import net.imagej.modelzoo.specification.DefaultOutputNodeSpecification;
import net.imagej.modelzoo.specification.DefaultTransformationSpecification;
import net.imagej.modelzoo.specification.InputNodeSpecification;
import net.imagej.modelzoo.specification.ModelSpecification;
import net.imagej.modelzoo.specification.OutputNodeSpecification;
import net.imagej.modelzoo.specification.TransformationSpecification;
import net.imglib2.type.numeric.real.FloatType;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DenoiSegModelSpecification extends DefaultModelSpecification {

	private final static String idTrainingKwargsTrainDimensions = "trainDimensions";
	private final static String idTrainingKwargsLearningRate = "learningRate";
	private final static String idTrainingKwargsNumEpochs = "numEpochs";
	private final static String idTrainingKwargsNumStepsPerEpoch = "numStepsPerEpoch";
	private final static String idTrainingKwargsBatchSize = "batchSize";
	private final static String idTrainingKwargsPatchShape = "patchShape";
	private final static String idTrainingKwargsNeighborhoodRadius = "neighborhoodRadius";
	private final static String idTrainingKwargsStepsFinished = "stepsFinished";
	private final static String idMean = "mean";
	private final static String idStdDev = "stdDev";
	private final static String citationText = "Tim-Oliver Buchholz and Mangal Prakash and Alexander Krull and Florian Jug. DenoiSeg: Joint Denoising and Segmentation. (2020)";
	private final static String doiText = "https://arxiv.org/abs/2005.02987";
	private final static List<String> tags = Arrays.asList("denoising", "segmentation", "unet2d");
	private final static String modelSource = "denoiseg";
	private final static String modelTrainingSource = DenoiSegTraining.class.getCanonicalName();
	private final static String modelInputName = DenoiSegTraining.predictionFeedInputOp;
	private final static String modelDataType = "float32";
	private final static List modelInputDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputDataRange = Arrays.asList("-inf", "inf");
	private final static String modelOutputName = DenoiSegTraining.predictionTargetOp;
	private final static String modelPreprocessing = DenoiSegPrediction.class.getCanonicalName() + "::preprocess";
	private final static String modelPostprocessing = DenoiSegPrediction.class.getCanonicalName() + "::postprocess";

	void writeModelConfigFile(DenoiSegConfig config, OutputHandler outputHandler, File targetDirectory, int stepsFinished) throws IOException {
		setMeta();
		setInputsOutputs(config);
		setTraining(config, stepsFinished);
		setPrediction(outputHandler);
		super.write(targetDirectory);
	}

	private void setPrediction(OutputHandler outputHandler) {
		TransformationSpecification preprocessing = new DefaultTransformationSpecification();
		preprocessing.setSpec(modelPreprocessing);
		Map<String, Object> normalizeArgs = new LinkedHashMap<>();
		normalizeArgs.put(idMean, Collections.singletonList(outputHandler.getMean().get()));
		normalizeArgs.put(idStdDev, Collections.singletonList(outputHandler.getStdDev().get()));
		preprocessing.setKwargs(new LinkedHashMap<>(normalizeArgs));
		addPredictionPreprocessing(preprocessing);
		TransformationSpecification postprocessing = new DefaultTransformationSpecification();
		postprocessing.setSpec(modelPostprocessing);
		postprocessing.setKwargs(new LinkedHashMap<>(normalizeArgs));
		addPredictionPostprocessing(postprocessing);
	}

	private void setTraining(DenoiSegConfig config, int stepsFinished) {
		setTrainingSource(modelTrainingSource);
		Map<String, Object> trainingKwargs = new LinkedHashMap<>();
		trainingKwargs.put(idTrainingKwargsBatchSize, config.getTrainBatchSize());
		trainingKwargs.put(idTrainingKwargsLearningRate, config.getLearningRate());
		trainingKwargs.put(idTrainingKwargsTrainDimensions, config.getTrainDimensions());
		trainingKwargs.put(idTrainingKwargsNeighborhoodRadius, config.getNeighborhoodRadius());
		trainingKwargs.put(idTrainingKwargsNumEpochs, config.getNumEpochs());
		trainingKwargs.put(idTrainingKwargsNumStepsPerEpoch, config.getStepsPerEpoch());
		trainingKwargs.put(idTrainingKwargsPatchShape, config.getTrainPatchShape());
		trainingKwargs.put(idTrainingKwargsStepsFinished, stepsFinished);
		setTrainingKwargs(trainingKwargs);
	}

	private void setInputsOutputs(DenoiSegConfig config) {
		List<Integer> modelInputMin;
		List<Integer> modelInputStep;
		List<Integer> modelInputHalo;
		List<Float> modelOutputScale;
		List<Integer> modelOutputOffset;
		String modelNodeAxes;
		int min = (int) Math.pow(2, config.getNetworkDepth());
		int halo = 96;
		if(config.getTrainDimensions() == 2) {
			modelNodeAxes = "byxc";
			modelInputMin = Arrays.asList(1, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, 0);
			modelOutputScale = Arrays.asList(1f, 1f, 1f, 1f);
			modelOutputOffset = Arrays.asList(0, 0, 0, 3);
		} else {
			modelNodeAxes = "bzyxc";
			modelInputMin = Arrays.asList(1, min, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, halo, 0);
			modelOutputScale = Arrays.asList(1f, 1f, 1f, 1f, 1f);
			modelOutputOffset = Arrays.asList(0, 0, 0, 0, 3);
		}
		InputNodeSpecification inputNode = new DefaultInputNodeSpecification();
		inputNode.setName(modelInputName);
		inputNode.setAxes(modelNodeAxes);
		inputNode.setDataType(modelDataType);
		inputNode.setDataRange(modelInputDataRange);
		inputNode.setHalo(modelInputHalo);
		inputNode.setShapeMin(modelInputMin);
		inputNode.setShapeStep(modelInputStep);
		addInputNode(inputNode);
		OutputNodeSpecification outputNode = new DefaultOutputNodeSpecification();
		outputNode.setName(modelOutputName);
		outputNode.setAxes(modelNodeAxes);
		outputNode.setDataType(modelDataType);
		outputNode.setDataRange(modelOutputDataRange);
		outputNode.setShapeReferenceInput(modelInputName);
		outputNode.setShapeScale(modelOutputScale);
		outputNode.setShapeOffset(modelOutputOffset);
		addOutputNode(outputNode);
	}

	private void setMeta() {
		CitationSpecification citation = new DefaultCitationSpecification();
		citation.setCitationText(citationText);
		citation.setDOIText(doiText);
		addCitation(citation);
		setTags(tags);
		setSource(modelSource);
	}

	public static boolean setFromSpecification(DenoiSegPrediction prediction, ModelSpecification specification) {
		double mean = 0.0f;
		double stdDev = 1.0f;

		List<TransformationSpecification> predictionPreprocessing = specification.getPredictionPreprocessing();
		if(predictionPreprocessing.size() > 0) {
			Map<String, Object> kwargs = predictionPreprocessing.get(0).getKwargs();
			if(kwargs != null) {
				List<? extends Number> meanObj = (List<? extends Number>) kwargs.get(idMean);
				if(meanObj != null && meanObj.size() > 0) mean = meanObj.get(0).doubleValue();
				List<? extends Number> stdDevObj = (List<? extends Number>) kwargs.get(idStdDev);
				if(stdDevObj != null && stdDevObj.size() > 0) stdDev = stdDevObj.get(0).doubleValue();
			}
		}

		prediction.setMean(new FloatType((float) mean));
		prediction.setStdDev(new FloatType((float) stdDev));

		return true;
	}

}
