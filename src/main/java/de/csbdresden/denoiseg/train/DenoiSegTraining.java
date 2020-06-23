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

import de.csbdresden.n2v.train.ModelZooTraining;
import de.csbdresden.n2v.train.RemainingTimeEstimator;
import de.csbdresden.n2v.ui.TrainingProgress;
import io.scif.services.DatasetIOService;
import net.imagej.ImageJ;
import net.imagej.modelzoo.consumer.model.tensorflow.TensorFlowConverter;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.thread.DefaultThreadService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.DialogPrompt;
import org.scijava.ui.UIService;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class DenoiSegTraining implements ModelZooTraining {

	private File graphDefFile;

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private OpService opService;

	@Parameter
	private UIService uiService;

	@Parameter
	private LogService logService;

	@Parameter
	private DatasetIOService datasetIOService;

	@Parameter
	private StatusService statusService;

	@Parameter
	private ThreadService threadService;

	@Parameter
	private Context context;

	// training feed
	private static final String trainingFeedXOp = "input";
	private static final String trainingFeedYOp = "activation_19_target";
	private static final String trainingFeedSampleWeightsOp = "activation_19_sample_weights";
	private static final String trainingFeedLearningPhaseOp = "keras_learning_phase";
	// training fetch
	private static final String trainingFetchLossOp = "loss/mul";
	private static final String trainingFetchDenoisegLossOp = "metrics/denoiseg/Mean_1";
	private static final String trainingFetchSegLossOp = "metrics/seg_loss/Mean_1";
	private static final String trainingFetchDenoiseLossOp = "metrics/denoise_loss/Mean";
	private static final String trainingFetchLearningOp = "Adam/lr/read";
	private static final String lrAssignOpName = "Adam/lr";
	// training target
	private static final String trainingTargetOp = "training/group_deps";

	// prediction feed
	static final String predictionFeedInputOp = trainingFeedXOp;
	// prediction target
	static final String predictionTargetOp = "activation_19/Identity";

	// validation target
	private static final String validationTargetOp = "group_deps";

	private TrainingProgress dialog;
	private PreviewHandler previewHandler;
	private OutputHandler outputHandler;
	private InputHandler inputHandler;

	private boolean stopTraining = false;

	private List<TrainingCallback> onEpochDoneCallbacks = new ArrayList<>();
	private List<TrainingCanceledCallback> onTrainingCanceled = new ArrayList<>();

	//TODO make setters etc.
	private boolean continueTraining = false;
	private File zipFile;
	private boolean canceled = false;
	private Session session;
	private DenoiSegConfig config;
	private int stepsFinished = 0;

	private int previewCount = 1;
	private int index;
	private Tensor<Float> tensorWeights;
	private XYPairs<FloatType> validationData;
	private List<Pair<Tensor, Tensor>> validationTensorData;
	private Future<?> future;

	public interface TrainingCallback {

		void accept(DenoiSegTraining training);
	}
	public interface TrainingCanceledCallback {

		void accept();
	}
	public DenoiSegTraining(Context context) {
		context.inject(this);
	}

	public void init(String trainedModel, DenoiSegConfig config) {
		if (Thread.interrupted()) return;
		continueTraining = true;
		zipFile = new File(trainedModel);
		init(config);
	}

	public void init(DenoiSegConfig config) {

		this.config = config;

		inputHandler= new InputHandler(context, config);

		if (Thread.interrupted()) return;

		if(!headless()) initDialog(config);

		if (Thread.interrupted()) return;

		logService.info( "Load TensorFlow.." );
		tensorFlowService.loadLibrary();
		logService.info( tensorFlowService.getStatus().getInfo() );

		addCallbackOnEpochDone(new ReduceLearningRateOnPlateau()::reduceLearningRateOnPlateau);
		addCallbackOnCancel(input()::cancel);

	}

	private void initDialog(DenoiSegConfig config) {
		dialog = TrainingProgress.create( this, config.getNumEpochs(), config.getStepsPerEpoch(), statusService, new DefaultThreadService() );
		dialog.setTitle("DenoiSeg");
		dialog.setWaitingIcon(getClass().getClassLoader().getResource( "bird.gif" ), 1, 1, 0, -10);
		inputHandler.setDialog(dialog);
		dialog.addTask( "Preparation" );
		dialog.addTask( "Training" );
		dialog.display();
		dialog.setTaskStart( 0 );
		dialog.setCurrentTaskMessage( "Loading TensorFlow" );

		//TODO warning if no GPU support
		//dialog.setWarning("WARNING: this will take for ever!");
	}

	private boolean headless() {
		return uiService.isHeadless();
	}

	public void train() {
		try {
			future = threadService.run(this::mainThread);
			future.get();
		} catch(CancellationException e) {
			if(stopTraining) return;
			cancel();
			logService.warn("DenoiSeg training canceled.");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	private void mainThread() {
		outputHandler = new OutputHandler(config, this, context);
		addCallbackOnEpochDone(outputHandler::copyBestModel);

		logTrainingStep("Create session..");
		if (Thread.interrupted() || isCanceled()) return;

		try (Graph graph = new Graph();
		     Session sess = new Session( graph )) {

			this.session = sess;

			loadGraph(graph);
			output().initTensors(sess);
			input().finalizeTrainingData();
			if(input().getTrainingData().size() == 0) {
				logService.error("Not training data available");
				return;
			}

			if (Thread.interrupted() || isCanceled()) return;
			logTrainingStep("Normalizing..");
			normalize();

			if (Thread.interrupted() || isCanceled()) return;
			logTrainingStep("Augment tiles..");
			augmentInputData();

			if (Thread.interrupted() || isCanceled()) return;
			logTrainingStep("Prepare training batches...");
			double n2v_perc_pix = 1.6;
			if (!batchNumSufficient(input().getTrainingData().size())) return;
			DenoiSegDataWrapper<FloatType> training_data = makeTrainingData(n2v_perc_pix);

			if (Thread.interrupted()) return;
			logTrainingStep("Prepare validation batches..");
			makeValidationData(n2v_perc_pix);

			index = 0;
			tensorWeights = makeWeightsTensor();

			if (handleInterruptionOrCancelation()) return;
			logTrainingStep("Start training..");
			if(!headless()) {
				threadService.queue(() -> {
					dialog.setTaskDone(0);
					dialog.setTaskStart(1);
				});
			}
			initPreviewHandler();

			RemainingTimeEstimator timeEstimator = initTimeEstimator();

			for (int epoch = 0; epoch < config().getNumEpochs() &&!stopTraining; epoch++) {
				updateTimeEstimator(timeEstimator, epoch);
				runEpoch(training_data, epoch);
				if (handleInterruptionOrCancelation()) return;
			}

//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			if ( !headless() ) {
				threadService.queue(() -> dialog.setTaskDone( 1 ));
			}
			stopTraining = true;
			logService.info( "Training done." );

//			if (inputs.size() > 0) uiService.show("inputs", Views.stack(inputs));
//			if (targets.size() > 0) uiService.show("targets", Views.stack(targets));

		}
		catch(IllegalStateException e) {
			cancel();
			e.printStackTrace();
			if(e.getMessage().contains("OOM")) {
				logService.error("Not enough memory available. Try to reduce the training batch size.");
			} else {
				e.printStackTrace();
			}
		} finally {
			if(tensorWeights != null) tensorWeights.close();
			if(validationTensorData != null) {
				for (Pair<Tensor, Tensor> pair : validationTensorData) {
					pair.getA().close();
					pair.getB().close();
				}
			}
		}
	}

//	public boolean confirmInputMatching(String title, File input1, File input2) {
//		InputConfirmationHandler inputConfirmationHandler = new InputConfirmationHandler(context, input1, input2);
//		return inputConfirmationHandler.confirmTrainingData();
//	}

	private boolean confirmInputData() {
		InputConfirmationHandler inputConfirmationHandler = new InputConfirmationHandler(context, input());
		boolean confirmed = inputConfirmationHandler.confirmTrainingData();
		if(!confirmed) return false;
		return inputConfirmationHandler.confirmValidationData();
	}

	private void augmentInputData() {
		DenoiSegDataGenerator.augment(input().getTrainingData());
		DenoiSegDataGenerator.augment(input().getValidationData());
	}

	private void runEpoch(DenoiSegDataWrapper<FloatType> training_data, int epoch) {
		List<Double> losses = new ArrayList<>(config().getStepsPerEpoch());
		for (int step = 0; step < config().getStepsPerEpoch() && !stopTraining; step++) {
			if (handleInterruptionOrCancelation()) return;
			runEpochStep(session, epoch, step, training_data, losses);
		}
		if (handleInterruptionOrCancelation()) return;
		training_data.on_epoch_end();
		float validationLoss = validate();
		if (handleInterruptionOrCancelation()) return;
		output().saveCheckpoint(session, previewHandler.getExampleInput(), previewHandler.getExampleOutput());
		output().setCurrentValidationSegLoss(validationLoss);
		if(!headless()) {
			threadService.queue(() -> dialog.updateTrainingChart(epoch + 1, losses, validationLoss));
		}
		onEpochDoneCallbacks.forEach(callback -> callback.accept(this));
	}

	private void logTrainingStep(String msg) {
		logService.info(msg);
		if (!headless()) {
			threadService.queue(() -> dialog.setCurrentTaskMessage(msg));
		}
	}

	private boolean handleInterruptionOrCancelation() {
		if (Thread.interrupted() || isCanceled()) {
			tensorWeights.close();
			return true;
		}
		return false;
	}

	private void initPreviewHandler() {
		previewHandler = new PreviewHandler(context, config().getTrainDimensions());
//		RandomAccessibleInterval<FloatType> second = validation_data.get(0).getSecond();
//		previewHandler.updateTrainingPreview(validation_data.get(0).getFirst(),
//				Views.interval(second, new FinalInterval(second.dimension(0), second.dimension(1), second.dimension(2), second.dimension(3)-1)));
//		previewHandler.updateValidationPreview(validation_data.get(0).getFirst(),
//				Views.interval(second, new FinalInterval(second.dimension(0), second.dimension(1), second.dimension(2), second.dimension(3)-1)));
	}

	private RemainingTimeEstimator initTimeEstimator() {
		RemainingTimeEstimator remainingTimeEstimator = new RemainingTimeEstimator();
		remainingTimeEstimator.setNumSteps(config().getNumEpochs());
		return remainingTimeEstimator;
	}

	private void updateTimeEstimator(RemainingTimeEstimator remainingTimeEstimator, int epoch) {
		remainingTimeEstimator.setCurrentStep(epoch);
		String remainingTimeString = remainingTimeEstimator.getRemainingTimeString();
		logService.info("Epoch " + (epoch + 1) + "/" + config().getNumEpochs() + " " + remainingTimeString);
	}

	private void loadGraph(Graph graph) {
		try {
			if(!continueTraining) {
				logService.info( "Import graph.." );
				output().loadUntrainedGraph(graph);
				output().createSavedModelDirs();
			}
			else {
				logService.info( "Import trained graph.." );
				File trainedModel = output().loadTrainedGraph(graph, zipFile);
				output().createSavedModelDirsFromExisting(trainedModel);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void runEpochStep(Session sess, int i, int j, DenoiSegDataWrapper<FloatType> training_data, List<Double> losses) {
		resetBatchIndexIfNeeded();
		Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item = training_data.getItem(index);
		runTrainingOp(sess, item);
		losses.add((double) output().getCurrentLoss());
		logStatusInConsole(j + 1, config().getStepsPerEpoch());
		if(!headless()) {
			threadService.queue(() -> dialog.updateTrainingProgress(i + 1, j + 1));
		}
		stepsFinished = config().getStepsPerEpoch()*i+j+1;
		index++;
	}

	private void resetBatchIndexIfNeeded() {
		if (index * config().getTrainBatchSize() + config().getTrainBatchSize() > input().getTrainingData().size() - 1) {
			index = 0;
			previewCount = 2;
			logService.info("starting with index 0 of training batches");
		}
	}

	private DenoiSegDataWrapper<FloatType> makeTrainingData(double n2v_perc_pix) {
		long[] patchShapeData = new long[config().getTrainDimensions()];
		Arrays.fill(patchShapeData, config().getTrainPatchShape());
		Dimensions patch_shape = new FinalDimensions(patchShapeData);

		return new DenoiSegDataWrapper<>(input().getTrainingData(), config().getTrainBatchSize(), n2v_perc_pix, patch_shape, config().getNeighborhoodRadius(), DenoiSegDataWrapper::uniform_withCP);
	}

	private void makeValidationData(double n2v_perc_pix) {
		int n_train = input().getTrainingData().size();
		int n_val = input().getValidationData().size();
		double frac_val = (1.0 * n_val) / (n_train + n_val);
		double frac_warn = 0.05;
		if (frac_val < frac_warn) {
			logService.info("small number of validation images (only " + (100 * frac_val) + "% of all images)");
		}
		long[] patchShapeData = new long[config().getTrainDimensions()];
		Arrays.fill(patchShapeData, config().getTrainPatchShape());
		Dimensions patch_shape = new FinalDimensions(patchShapeData);
		DenoiSegDataWrapper<FloatType> valData = new DenoiSegDataWrapper<>(input().getValidationData(),
				Math.min(config().getTrainBatchSize(), input().getValidationData().size()),
				n2v_perc_pix, patch_shape, config().getNeighborhoodRadius(),
				DenoiSegDataWrapper::uniform_withCP);

		XYPairs<FloatType> validationDataList = new XYPairs<>();
		for (int i = 0; i < valData.numBatches(); i++) {
			validationDataList.add(valData.getItem(i));
		}
		this.validationData = validationDataList;
		validationTensorData = new ArrayList<>();
		for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> pair : validationDataList) {
			Tensor tensorX = TensorFlowConverter.imageToTensor(pair.getA(), getMapping());
			Tensor tensorY = TensorFlowConverter.imageToTensor(pair.getB(), getMapping());
			validationTensorData.add(new ValuePair<>(tensorX, tensorY));
		}
	}

	private void normalize() {
		FloatType mean = output().getMean();
		FloatType stdDev = output().getStdDev();
		List<RandomAccessibleInterval<FloatType>> x = new ArrayList<>();
		for (Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> pair : input().getTrainingData()) {
			x.add(pair.getA());
		}
		mean.set( opService.stats().mean( Views.iterable( Views.stack(x) ) ).getRealFloat() );
		stdDev.set( opService.stats().stdDev( Views.iterable( Views.stack(x) ) ).getRealFloat() );
		logService.info("mean: " + mean.get());
		logService.info("stdDev: " + stdDev.get());

		TrainUtils.normalize( input().getTrainingData(), mean, stdDev );
		TrainUtils.normalize( input().getValidationData(), mean, stdDev );
	}

	private void runTrainingOp(Session sess, Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item) {
//		if(previewCount-- > 0) {
//			opService.context().service(UIService.class).show("X", item.getA());
//			opService.context().service(UIService.class).show("Y", item.getB());
//		}
//		System.out.println("X: " + Arrays.toString(Intervals.dimensionsAsIntArray(item.getFirst())));
//		System.out.println("Y: " + Arrays.toString(Intervals.dimensionsAsIntArray(item.getSecond())));
		Tensor tensorX = TensorFlowConverter.imageToTensor(item.getA(), getMapping());
		Tensor tensorY = TensorFlowConverter.imageToTensor(item.getB(), getMapping());

		Session.Runner runner = sess.runner();

		Tensor<Float> learningRate = Tensors.create(output().getCurrentLearningRate());
		Tensor<Boolean> learningPhase = Tensors.create(true);
		runner.feed(trainingFeedXOp, tensorX).feed(trainingFeedYOp, tensorY)
				.feed(trainingFeedLearningPhaseOp, learningPhase)
				.feed(lrAssignOpName, learningRate)
				.feed(trainingFeedSampleWeightsOp, tensorWeights).addTarget(trainingTargetOp);
		runner.fetch(trainingFetchLossOp);
		runner.fetch(trainingFetchDenoisegLossOp);
		runner.fetch(trainingFetchDenoiseLossOp);
		runner.fetch(trainingFetchSegLossOp);
		runner.fetch(trainingFetchLearningOp);

		List<Tensor<?>> fetchedTensors = runner.run();
		output().setCurrentLoss(fetchedTensors.get(0).floatValue());
		output().setCurrentDenoisegLoss(fetchedTensors.get(1).floatValue());
		output().setCurrentDenoiseLoss(fetchedTensors.get(2).floatValue());
		output().setCurrentSegLoss(fetchedTensors.get(3).floatValue());
		output().setCurrentLearningRate(fetchedTensors.get(4).floatValue());

		fetchedTensors.forEach(Tensor::close);
		tensorX.close();
		tensorY.close();
		learningPhase.close();
		learningRate.close();
	}

	public void addCallbackOnEpochDone(TrainingCallback callback) {
		onEpochDoneCallbacks.add(callback);
	}

	private int[] getMapping() {
		if(config().getTrainDimensions() == 2) return new int[]{ 1, 2, 0, 3 };
		if(config().getTrainDimensions() == 3) return new int[]{ 1, 2, 3, 0, 4 };
		return new int[0];
	}

	private boolean batchNumSufficient(int n_train) {
		if(config().getTrainBatchSize() > n_train) {
			String errorMsg = "Not enough training data (" + n_train + " batches). At least " + config().getTrainBatchSize() + " batches needed.";
			logService.error(errorMsg);
			stopTraining = true;
			dispose();
			uiService.showDialog(errorMsg, DialogPrompt.MessageType.ERROR_MESSAGE);
			return false;
		}
		return true;
	}

	private Tensor<Float> makeWeightsTensor() {
		float[] weightsdata = new float[config().getTrainBatchSize()];
		Arrays.fill(weightsdata, 1);
		return Tensors.create(weightsdata);
	}

	private float validate() {

		float avgDenoiseLoss = 0;
		float avgSegLoss = 0;

		long validationBatches = validationTensorData.size();
		int i = 0;
		for (; i < validationBatches; i++) {

			Pair<Tensor, Tensor> tensorItem = validationTensorData.get(i);
			Pair<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> item = validationData.get(i);

			Tensor tensorX = tensorItem.getA();
			Tensor tensorY = tensorItem.getB();
			Tensor<Boolean> tensorLearningPhase = Tensors.create(false);

			Session.Runner runner = session.runner();

			runner.feed(trainingFeedXOp, tensorX)
					.feed(trainingFeedYOp, tensorY)
					.feed(trainingFeedLearningPhaseOp, tensorLearningPhase)
					.feed(trainingFeedSampleWeightsOp, tensorWeights)
					.addTarget(validationTargetOp);
			runner.fetch(trainingFetchDenoiseLossOp);
			runner.fetch(trainingFetchSegLossOp);
			if(!headless() && i == 0) runner.fetch(predictionTargetOp);

			List<Tensor<?>> fetchedTensors = runner.run();

			avgDenoiseLoss += fetchedTensors.get(0).floatValue();
			avgSegLoss += fetchedTensors.get(1).floatValue();

			if(!headless() && i == 0) {
				Tensor outputTensor = fetchedTensors.get(2);
				RandomAccessibleInterval<FloatType> output = TensorFlowConverter.tensorToImage(outputTensor, getMapping());
				previewHandler.updateValidationPreview(item.getA(), output, headless());
//			updateHistoryImage(output);
			}
			fetchedTensors.forEach(Tensor::close);
			tensorLearningPhase.close();

			if (stopTraining || Thread.interrupted() || isCanceled()) {
				break;
			}
		}
		avgDenoiseLoss /= (float)(i+1);
		avgSegLoss /= (float)(i+1);

		logService.info("\nValidation denoise loss: " + avgDenoiseLoss + " seg loss: " + avgSegLoss);
		return avgDenoiseLoss;
	}

	public int getStepsFinished() {
		return stepsFinished;
	}

	public void setLearningRate(float newLR) {
		output().setCurrentLearningRate(newLR);
	}

	private void logStatusInConsole(int step, int stepTotal) {
		int maxBareSize = 10; // 10unit for 100%
		int remainProcent = ( ( 100 * step ) / stepTotal ) / maxBareSize;
		char defaultChar = '-';
		String icon = "*";
		String bare = new String( new char[ maxBareSize ] ).replace( '\0', defaultChar ) + "]";
		StringBuilder bareDone = new StringBuilder();
		bareDone.append( "[" );
		for ( int i = 0; i < remainProcent; i++ ) {
			bareDone.append( icon );
		}
		String bareRemain = bare.substring( remainProcent );
		System.out.printf( "%d / %d %s%s - loss: %f denoiseg loss: %f seg loss: %f denoise loss: %f lr: %f\n", step, stepTotal, bareDone, bareRemain,
				output().getCurrentLoss(),
				output().getCurrentDenoiSegLoss(),
				output().getCurrentSegLoss(),
				output().getCurrentDenoiseLoss(),
				output().getCurrentLearningRate() );
	}

	public TrainingProgress getDialog() {
		return dialog;
	}

	public DenoiSegConfig config() {
		return config;
	}

	public InputHandler input() {
		return inputHandler;
	}

	public OutputHandler output() {
		return outputHandler;
	}

	@Override
	public void stopTraining() {
		stopTraining = true;
	}

	@Override
	public void cancel() {
		canceled = true;
		onTrainingCanceled.forEach(TrainingCanceledCallback::accept);
		if(future != null) future.cancel(true);
		dispose();
		if(getDialog() != null) getDialog().dispose();
	}

	public boolean isCanceled() {
		return canceled;
	}

	public void addCallbackOnCancel(TrainingCanceledCallback callback) {
		onTrainingCanceled.add(callback);
	}

	public void dispose() {
		if(output() != null) output().dispose();
	}

	public static void main( final String... args ) throws Exception {

		final ImageJ ij = new ImageJ();
		ij.launch( args );

		File trainX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_train");
		File trainY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_train");
		File valX = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/X_val");
		File valY = new File("/home/random/Development/imagej/project/CSBDeep/data/DenoiSeg/data/DSB/train_data/10/Y_val");

		DenoiSegTraining training = new DenoiSegTraining(ij.context());
		training.init(new DenoiSegConfig()
				.setNumEpochs(200)
				.setStepsPerEpoch(2)
				.setBatchSize(32)
				.setPatchShape(64));
		training.input().addTrainingData(trainX, trainY);
		training.input().addValidationData(valX, valY);
		training.train();
		if(!training.isCanceled()) {
			File modelFile = training.output().exportLatestTrainedModel();
			Object model = ij.io().open(modelFile.getAbsolutePath());
			ij.ui().show(model);
		}
	}
}
