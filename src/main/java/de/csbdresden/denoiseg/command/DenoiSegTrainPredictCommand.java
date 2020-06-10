package de.csbdresden.denoiseg.command;

import de.csbdresden.denoiseg.train.DenoiSegConfig;
import de.csbdresden.denoiseg.predict.N2VPrediction;
import de.csbdresden.denoiseg.train.DenoiSegTraining;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.logic.BoolType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Cancelable;
import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.io.IOService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Plugin( type = Command.class, menuPath = "Plugins>CSBDeep>N2V>train + predict" )
public class DenoiSegTrainPredictCommand implements Command, Cancelable {


	@Parameter(label = "Folder containing training raw images")
	private File trainingRawData;

	@Parameter(label = "Folder containing training labeling images")
	private File trainingLabelingData;

	@Parameter(label = "Folder containing validation raw images")
	private File validationRawData;

	@Parameter(label = "Folder containing validation labeling images")
	private File validationLabelingData;

	@Parameter(label = "Raw prediction input image")
	private Img predictionInput;

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

	@Parameter(label = "Batch size per step")
	private int batchSize = 180;

	@Parameter(label = "Dimension length of batch")
	private int batchDimLength = 180;

	@Parameter(label = "Dimension length of patch")
	private int patchDimLength = 60;

	@Parameter(label = "Neighborhood radius")
	private int neighborhoodRadius = 0;

	@Parameter( type = ItemIO.OUTPUT )
	private RandomAccessibleInterval< FloatType > output;

	@Parameter(type = ItemIO.OUTPUT, label = "model from last training step")
	private String latestTrainedModelPath;

	@Parameter(type = ItemIO.OUTPUT, label = "model with lowest validation loss")
	private String bestTrainedModelPath;

	@Parameter
	private CommandService commandService;

	@Parameter
	private OpService opService;

	@Parameter
	private Context context;

	@Parameter
	private LogService logService;

	@Parameter
	private IOService ioService;

	private boolean canceled = false;

	private ExecutorService pool;
	private Future<?> future;
	private DenoiSegTraining training;
	@Override
	public void run() {

		pool = Executors.newSingleThreadExecutor();

		try {

			future = pool.submit(this::mainThread);
			future.get();

		} catch(CancellationException e) {
			logService.warn("N2V train + predict command canceled.");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	private void mainThread() {

		training = new DenoiSegTraining(context);
		training.addCallbackOnCancel(this::cancel);
		training.init(new DenoiSegConfig()
				.setNumEpochs(numEpochs)
				.setStepsPerEpoch(numStepsPerEpoch)
				.setBatchSize(batchSize)
				.setBatchDimLength(batchDimLength)
				.setPatchDimLength(patchDimLength)
				.setNeighborhoodRadius(neighborhoodRadius));
		if(training.getDialog() != null) training.getDialog().addTask( "Prediction" );

		try {
			if(validationRawData.equals(trainingRawData)) {
				System.out.println("Using 10% of training data for validation");
				training.input().addTrainingAndValidationData(trainingRawData, trainingLabelingData);
			} else {
				training.input().addTrainingData(trainingRawData, trainingLabelingData);
				training.input().addValidationData(validationRawData, validationLabelingData);
			}
			training.train();
			if(training.isCanceled()) cancel("");
		}
		catch(Exception e) {
			training.dispose();
			e.printStackTrace();
			return;
		}
		try {
			File savedModel = training.output().exportLatestTrainedModel();
			if(savedModel == null) return;
			latestTrainedModelPath = savedModel.getAbsolutePath();
			savedModel = training.output().exportBestTrainedModel();
			bestTrainedModelPath = savedModel.getAbsolutePath();
		} catch (IOException e) {
			e.printStackTrace();
		}

		if(isCanceled()) return;

		if(training.getDialog() != null) training.getDialog().setTaskStart(2);

		if(latestTrainedModelPath == null) return;

		N2VPrediction prediction = new N2VPrediction(context);
		prediction.setModelFile(new File(latestTrainedModelPath));
		this.output = prediction.predictPadded(this.predictionInput);

		if(training.getDialog() != null) training.getDialog().setTaskDone(2);

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
		if(future != null) {
			future.cancel(true);
		}
		if(pool != null) {
			pool.shutdownNow();
		}
	}

	@Override
	public String getCancelReason() {
		return null;
	}
	
	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );

		File trainX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train");
		File trainY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_train");
		File valX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_val");
		File valY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_val");

		final String predictionFile = "/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train/img_0.tif";

		RandomAccessibleInterval prediction = ( RandomAccessibleInterval ) ij.io().open( predictionFile );

		ij.command().run( DenoiSegTrainPredictCommand.class, true,
				"trainingRawData", trainX,
				"trainingLabelingData", trainY,
				"validationRawData", valX,
				"validationLabelingData", valY,
				"predictionInput", prediction).get();
	}
}
