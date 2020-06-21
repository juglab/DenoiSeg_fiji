package de.csbdresden.denoiseg.command;

import de.csbdresden.denoiseg.predict.DenoiSegPrediction;
import io.scif.MissingLibraryException;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>DenoiSeg>DenoiSeg train + predict" )
public class DenoiSegTrainPredictCommand extends DenoiSegTrainCommand {

	@Parameter(required = false, visibility = ItemVisibility.MESSAGE)
	private String predictionLabel = "<html><br/><span style='font-weight: normal'>Prediction</span></html>";

	@Parameter(label = "Raw prediction input image")
	private Img predictionInput;

	@Parameter(label = "Axes of prediction input (subset of XYB, B = batch")
	private String axes = "XY";

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	private boolean canceled = false;

	@Override
	public void run() {

		System.out.println("Launching the DenoiSeg prediction command");

		try {
			System.out.println("Launching the DenoiSeg training command");

			File latestModel = train();
			if(latestModel == null) return;
			if(training.getDialog() != null) training.getDialog().setTaskStart(2);
			openSavedModels(latestModel);
			predict();
			if(training.getDialog() != null) training.getDialog().setTaskDone(2);
		} catch (IOException | MissingLibraryException e) {
			e.printStackTrace();
			training.getDialog().dispose();
		} finally {
			training.dispose();
		}

	}

	private void predict() throws FileNotFoundException, MissingLibraryException {
		DenoiSegPrediction prediction = new DenoiSegPrediction(context);
		prediction.setTrainedModel(latestTrainedModel);
		this.output = prediction.predictPadded(this.predictionInput, axes);
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

		ij.command().run( DenoiSegTrainPredictCommand.class, true,
				"trainingRawData", trainX,
				"trainingLabelingData", trainY,
				"validationRawData", valX,
				"validationLabelingData", valY,
				"predictionInput", ij.io().open(predictionFile)).get();
	}
}
