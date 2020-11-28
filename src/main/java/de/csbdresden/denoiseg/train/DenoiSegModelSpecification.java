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

import io.bioimage.specification.CitationSpecification;
import io.bioimage.specification.DefaultCitationSpecification;
import io.bioimage.specification.DefaultInputNodeSpecification;
import io.bioimage.specification.DefaultOutputNodeSpecification;
import io.bioimage.specification.InputNodeSpecification;
import io.bioimage.specification.OutputNodeSpecification;
import io.bioimage.specification.WeightsSpecification;
import io.bioimage.specification.transformation.ClipTransformation;
import io.bioimage.specification.transformation.ImageTransformation;
import io.bioimage.specification.transformation.ScaleLinearTransformation;
import io.bioimage.specification.transformation.ZeroMeanUnitVarianceTransformation;
import io.bioimage.specification.weights.TensorFlowSavedModelBundleSpecification;
import net.imagej.modelzoo.specification.ImageJModelSpecification;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DenoiSegModelSpecification extends ImageJModelSpecification {

	private final static String idTrainingKwargsTrainDimensions = "trainDimensions";
	private final static String idTrainingKwargsLearningRate = "learningRate";
	private final static String idTrainingKwargsNumEpochs = "numEpochs";
	private final static String idTrainingKwargsNumStepsPerEpoch = "numStepsPerEpoch";
	private final static String idTrainingKwargsBatchSize = "batchSize";
	private final static String idTrainingKwargsPatchShape = "patchShape";
	private final static String idTrainingKwargsNeighborhoodRadius = "neighborhoodRadius";
	private final static String idTrainingKwargsStepsFinished = "stepsFinished";
	private final static String citationText = "Tim-Oliver Buchholz and Mangal Prakash and Alexander Krull and Florian Jug. DenoiSeg: Joint Denoising and Segmentation. (2020)";
	private final static String doiText = "https://arxiv.org/abs/2005.02987";
	private final static List<String> tags = Arrays.asList("denoising", "segmentation", "unet2d");
	private final static String modelSource = "denoiseg";
	private final static String modelTrainingSource = DenoiSegTraining.class.getCanonicalName();
	private final static String modelInputName = DenoiSegTraining.predictionFeedInputOp;
	private final static String modelDataType = "float32";
	private final static List modelInputDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputDenoiseDataRange = Arrays.asList("-inf", "inf");
	private final static List modelOutputSegmentDataRange = Arrays.asList(0, 1);
	private final static String modelOutputSegmentName = DenoiSegTraining.predictionTargetSegmentOp;
	private final static String modelOutputDenoiseName = DenoiSegTraining.predictionTargetDenoiseOp;

	DenoiSegModelSpecification() {
		super();
	}

	void update(DenoiSegConfig config, DenoiSegOutputHandler outputHandler, int stepsFinished) {
		setMeta(outputHandler);
		setInputsOutputs(config, outputHandler);
		setTraining(config, stepsFinished);
		setWeights(outputHandler);
	}

	private void setWeights(DenoiSegOutputHandler outputHandler) {
		WeightsSpecification weights = new TensorFlowSavedModelBundleSpecification();
		weights.setSource(outputHandler.getSavedModelBundlePackage());
		addWeights(weights);
	}

	private void setTraining(DenoiSegConfig config, int stepsFinished) {
		String trainingSource = modelTrainingSource;
		Map<String, Object> trainingKwargs = new LinkedHashMap<>();
		trainingKwargs.put(idTrainingKwargsBatchSize, config.getTrainBatchSize());
		trainingKwargs.put(idTrainingKwargsLearningRate, config.getLearningRate());
		trainingKwargs.put(idTrainingKwargsTrainDimensions, config.getTrainDimensions());
		trainingKwargs.put(idTrainingKwargsNeighborhoodRadius, config.getNeighborhoodRadius());
		trainingKwargs.put(idTrainingKwargsNumEpochs, config.getNumEpochs());
		trainingKwargs.put(idTrainingKwargsNumStepsPerEpoch, config.getStepsPerEpoch());
		trainingKwargs.put(idTrainingKwargsPatchShape, config.getTrainPatchShape());
		trainingKwargs.put(idTrainingKwargsStepsFinished, stepsFinished);
		setTrainingStats(trainingSource, trainingKwargs);
	}

	private void setInputsOutputs(DenoiSegConfig config, DenoiSegOutputHandler outputHandler) {
		List<Integer> modelInputMin;
		List<Integer> modelInputStep;
		List<Integer> modelInputHalo;
		List<Float> modelInputScale;
		List<Float> modelOutputScaleDenoise;
		List<Float> modelOutputScaleSegment;
		List<Integer> modelOutputOffsetDenoise;
		List<Integer> modelOutputOffsetSegment;
		String modelNodeAxes;
		int min = (int) Math.pow(2, config.getNetworkDepth());
		int halo = 96;
		if(config.getTrainDimensions() == 2) {
			modelNodeAxes = "byxc";
			modelInputMin = Arrays.asList(1, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, 0);
			modelOutputScaleDenoise = Arrays.asList(1f, 1f, 1f, 1f);
			modelOutputScaleSegment = Arrays.asList(1f, 1f, 1f, 1f);
			modelOutputOffsetDenoise = Arrays.asList(0, 0, 0, 0);
			modelOutputOffsetSegment = Arrays.asList(0, 0, 0, 2);
		} else {
			modelNodeAxes = "bzyxc";
			modelInputMin = Arrays.asList(1, min, min, min, 1);
			modelInputStep = Arrays.asList(1, min, min, min, 0);
			modelInputHalo = Arrays.asList(0, halo, halo, halo, 0);
			modelOutputScaleDenoise = Arrays.asList(1f, 1f, 1f, 1f, 1f);
			modelOutputScaleSegment = Arrays.asList(1f, 1f, 1f, 1f, 1f);
			modelOutputOffsetDenoise = Arrays.asList(0, 0, 0, 0, 0);
			modelOutputOffsetSegment = Arrays.asList(0, 0, 0, 0, 2);
		}
		InputNodeSpecification inputNode = new DefaultInputNodeSpecification();
		inputNode.setName(modelInputName);
		inputNode.setAxes(modelNodeAxes);
		inputNode.setDataType(modelDataType);
		inputNode.setDataRange(modelInputDataRange);
		inputNode.setHalo(modelInputHalo);
		inputNode.setShapeMin(modelInputMin);
		inputNode.setShapeStep(modelInputStep);
		ZeroMeanUnitVarianceTransformation preprocessing = new ZeroMeanUnitVarianceTransformation();
		preprocessing.setMode(ImageTransformation.Mode.FIXED);
		preprocessing.setMean(outputHandler.getMean().get());
		preprocessing.setStd(outputHandler.getStdDev().get());
		inputNode.setPreprocessing(Collections.singletonList(preprocessing));
		addInputNode(inputNode);
		OutputNodeSpecification denoiseOutput = new DefaultOutputNodeSpecification();
		denoiseOutput.setName(modelOutputDenoiseName);
		denoiseOutput.setAxes(modelNodeAxes);
		denoiseOutput.setDataType(modelDataType);
		denoiseOutput.setDataRange(modelOutputDenoiseDataRange);
		denoiseOutput.setShapeReferenceInput(modelInputName);
		denoiseOutput.setShapeScale(modelOutputScaleDenoise);
		denoiseOutput.setShapeOffset(modelOutputOffsetDenoise);
		ScaleLinearTransformation postprocessing = new ScaleLinearTransformation();
		postprocessing.setMode(ImageTransformation.Mode.FIXED);
		postprocessing.setOffset(outputHandler.getMean().get());
		postprocessing.setGain(outputHandler.getStdDev().get());
		denoiseOutput.setPostprocessing(Collections.singletonList(postprocessing));
		addOutputNode(denoiseOutput);
		OutputNodeSpecification segmentOutput = new DefaultOutputNodeSpecification();
		segmentOutput.setName(modelOutputSegmentName);
		segmentOutput.setAxes(modelNodeAxes);
		segmentOutput.setDataType(modelDataType);
		segmentOutput.setDataRange(modelOutputDenoiseDataRange);
		segmentOutput.setShapeReferenceInput(modelInputName);
		segmentOutput.setShapeScale(modelOutputScaleSegment);
		segmentOutput.setShapeOffset(modelOutputOffsetSegment);
		ClipTransformation clip = new ClipTransformation();
		clip.setMode(ImageTransformation.Mode.FIXED);
		clip.setMin(0);
		clip.setMax(1);
		segmentOutput.setPostprocessing(Collections.singletonList(clip));
		addOutputNode(segmentOutput);
	}

	private void setMeta(DenoiSegOutputHandler outputHandler) {
		CitationSpecification citation = new DefaultCitationSpecification();
		citation.setCitationText(citationText);
		citation.setDOIText(doiText);
		addCitation(citation);
		setTags(tags);
		setSource(modelSource);
		setSampleInputs(outputHandler.getSampleInputNames());
		setSampleOutputs(outputHandler.getSampleOutputNames());
	}

}
