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
package de.csbdresden.denoiseg.command;

import de.csbdresden.denoiseg.train.DenoiSegConfig;
import de.csbdresden.denoiseg.train.DenoiSegTraining;
import net.imagej.ImageJ;
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.NumberWidget;

import java.io.File;
import java.io.IOException;

import static org.scijava.widget.FileWidget.DIRECTORY_STYLE;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>DenoiSeg>DenoiSeg train" )
public class DenoiSegTrainCommand implements Command, Cancelable {

	@Parameter(label = "Folder containing training raw images", style = DIRECTORY_STYLE)
	private File trainingRawData;

	@Parameter(label = "Folder containing training labeling images", style = DIRECTORY_STYLE)
	private File trainingLabelingData;

	@Parameter(label = "Folder containing validation raw images", style = DIRECTORY_STYLE)
	private File validationRawData;

	@Parameter(label = "Folder containing validation labeling images", style = DIRECTORY_STYLE)
	private File validationLabelingData;

	//TODO make these parameters work
//	@Parameter(label = "Training mode", choices = {"start new training", "continue training"})
//	private String trainBase;
//
//	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
//	private String newTrainingLabel = "<html><br/><span style='font-weight: normal'>Options for new training</span></html>";

//	@Parameter(label = "Use 3D model instead of 2D")
//	private boolean mode3D = false;

	//TODO make these parameters work
//	@Parameter(label = "Start from model trained on noise")
//	private boolean startFromNoise = false;
//
//	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
//	private String continueTrainingLabel = "<html><br/><span style='font-weight: normal'>Options for continuing training</span></html>";
//
//	@Parameter(required = false, label = "Pretrained model file (.zip)")
//	private File pretrainedNetwork;

	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
	private String advancedLabel = "<html><br/><span style='font-weight: normal'>Advanced options</span></html>";

	@Parameter(label = "Number of epochs")
	private int numEpochs = 300;

	@Parameter(label = "Number of steps per epoch")
	private int numStepsPerEpoch = 200;

	@Parameter(label = "Batch size")
	private int batchSize = 64;

	@Parameter(label = "Patch shape", min = "16", max = "512", stepSize = "16", style= NumberWidget.SLIDER_STYLE)
	private int patchShape = 64;

	@Parameter(label = "Neighborhood radius")
	private int neighborhoodRadius = 5;

	@Parameter(type = ItemIO.OUTPUT, label = "Model from last training step")
	protected ModelZooArchive latestTrainedModel;

	@Parameter(type = ItemIO.OUTPUT, label = "Model with lowest validation loss")
	private ModelZooArchive bestTrainedModel;

	@Parameter
	protected Context context;

	@Parameter
	private ModelZooService modelZooService;

	private boolean canceled = false;
	protected DenoiSegTraining training;

	@Override
	public void run() {

		System.out.println("Launching the DenoiSeg training command");

		try {
			File savedModel = train();
			if(savedModel == null) return;
			openSavedModels(savedModel);
		} catch (IOException e) {
			e.printStackTrace();
			training.getDialog().dispose();
		} finally {
			training.dispose();
		}

	}

	protected void openSavedModels(File savedModel) throws IOException {
		latestTrainedModel = modelZooService.open(savedModel);
		savedModel = training.output().exportBestTrainedModel();
		bestTrainedModel = modelZooService.open(savedModel);
	}

	protected File train() throws IOException {
		training = new DenoiSegTraining(context);
		training.addCallbackOnCancel(this::cancel);
		training.init(new DenoiSegConfig()
				.setNumEpochs(numEpochs)
				.setStepsPerEpoch(numStepsPerEpoch)
				.setBatchSize(batchSize)
				.setPatchShape(patchShape)
				.setNeighborhoodRadius(neighborhoodRadius));
		if(training.getDialog() != null) training.getDialog().addTask( "Prediction" );

//		training.confirmInputMatching("training", trainingRawData, trainingLabelingData);

		if(trainingRawData.getAbsolutePath().equals(validationRawData.getAbsolutePath()) &&
				trainingLabelingData.getAbsolutePath().equals(validationLabelingData.getAbsolutePath())) {
			training.input().addTrainingAndValidationData(trainingRawData, trainingLabelingData);
		} else {
			training.input().addTrainingData(trainingRawData, trainingLabelingData);
			training.input().addValidationData(validationRawData, validationLabelingData);
		}
		training.train();
		if(training.isCanceled()) cancel("");
		return training.output().exportLatestTrainedModel();
	}

	private void cancel() {
		cancel("");
	}

	@Override
	public boolean isCanceled() {
		return canceled;
	}

	@Override
	public void cancel(String reason) {
		canceled = true;
		if(training != null) training.dispose();
	}

	@Override
	public String getCancelReason() {
		return null;
	}
	
	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch(args);

		File trainX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train");
		File trainY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_train");
		File valX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_val");
		File valY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_val");

		ij.command().run( DenoiSegTrainCommand.class, true,
				"trainingRawData", trainX,
				"trainingLabelingData", trainY,
				"validationRawData", valX,
				"validationLabelingData", valY,
//				"batchSize", 32,
				"patchShape", 64,
				"numStepsPerEpoch", 200,
				"numEpochs", 200).get();
	}
}
