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

import de.csbdresden.denoiseg.predict.DenoiSegPrediction;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>DenoiSeg>DenoiSeg train + predict" )
public class DenoiSegTrainPredictCommand extends DenoiSegTrainCommand {

	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
	private String predictionLabel = "<html><br/><span style='font-weight: normal'>Prediction</span></html>";

	@Parameter(label = "Raw prediction input image")
	private RandomAccessibleInterval<FloatType> predictionInput;

	@Parameter(label = "Axes of prediction input (subset of XYB, B = batch)")
	private String axes = "XY";

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	private boolean canceled = false;

	@Override
	public void run() {

		System.out.println("Launching the DenoiSeg train & predict command");

		try {

			File latestModel = train();
			if(latestModel == null) return;
			if(training.getDialog() != null) training.getDialog().setTaskStart(2);
			openSavedModels(latestModel);
			predict();
			if(training.getDialog() != null) training.getDialog().setTaskDone(2);
		} catch (Exception e) {
			e.printStackTrace();
			training.getDialog().dispose();
		} finally {
			training.dispose();
		}

	}

	private void predict() throws Exception {
		DenoiSegPrediction prediction = new DenoiSegPrediction(context);
		prediction.setTrainedModel(latestTrainedModel);
		this.output = prediction.predict(this.predictionInput, axes);
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

		File trainX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/X_train");
		File trainY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/Y_train");
		File valX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/X_val");
		File valY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/Y_val");

		final String predictionFile = "/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/mouse/Mouse_n10/X_test/img_1.tif";

		Object img = ij.io().open(predictionFile);
		ij.ui().show(img);
		CommandModule res = ij.command().run(DenoiSegTrainPredictCommand.class, false,
				"trainingRawData", trainX,
				"trainingLabelingData", trainY,
				"validationRawData", valX,
				"validationLabelingData", valY,
				"predictionInput", img).get();
		ij.ui().show(res.getOutput("output"));
	}
}
