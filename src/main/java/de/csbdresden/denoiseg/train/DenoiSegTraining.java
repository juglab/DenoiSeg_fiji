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
import net.imagej.modelzoo.ModelZooArchive;
import net.imagej.modelzoo.ModelZooService;
import net.imagej.modelzoo.consumer.model.tensorflow.TensorFlowConverter;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.IntervalView;
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


	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private OpService opService;

	@Parameter
	private UIService uiService;

	@Parameter
	private LogService logService;

	@Parameter
	private StatusService statusService;

	@Parameter
	private ThreadService threadService;

	@Parameter
	private ModelZooService modelZooService;

	@Parameter
	private Context context;

	// training feed
	private static final String trainingFeedXOp = "input";
	private static final String trainingFeedYDenoiseOp = "out_denoise_target";
	private static final String trainingFeedYSegmentOp = "out_segment_target";
	private static final String trainingFeedSampleWeightsDenoiseOp = "out_denoise_sample_weights";
	private static final String trainingFeedSampleWeightsSegmentOp = "out_segment_sample_weights";
	private static final String trainingFeedLearningPhaseOp = "keras_learning_phase";
	// training fetch
	private static final String trainingFetchLossOp = "loss_tensor";
	private static final String trainingFetchSegLossOp = "out_segment_loss_tensor";
	private static final String trainingFetchDenoiseLossOp = "out_denoise_loss_tensor";
	private static final String trainingFetchLearningOp = "read_learning_rate";
	private static final String lrAssignOpName = "write_learning_rate";
	// training target
	private static final String trainingTargetOp = "train";
	// prediction feed
	static final String predictionFeedInputOp = trainingFeedXOp;
	// prediction target
	static final String predictionTargetDenoiseOp = "denoised";
	static final String predictionTargetSegmentOp = "segmented";
	// validation target
	private static final String validationTargetOp = "validation";

	private TrainingProgress dialog;

	private PreviewHandler previewHandler;
	private DenoiSegOutputHandler outputHandler;
	private InputHandler inputHandler;
	private boolean stopTraining = false;

	private List<TrainingCallback> onEpochDoneCallbacks = new ArrayList<>();
	private List<TrainingCanceledCallback> onTrainingCanceled = new ArrayList<>();

	private boolean continueTraining = false;
	private File zipFile;
	private boolean canceled = false;
	private Session session;
	private DenoiSegConfig config;
	private int stepsFinished = 0;
	private int previewCount = 1;

	private int index;
	private Tensor<Float> tensorWeightsSegment;
	private Tensor<Float> tensorWeightsDenoise;
	private ProcessedTrainingDataCollection<FloatType> validationData;
	private List<Pair<Tensor, Pair<Tensor, Tensor>>> validationTensorData;
	private Future<?> future;
	private int count = 0;

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

		inputHandler = new InputHandler(context, config);

		if (Thread.interrupted()) return;

		if (!headless()) initDialog(config);

		if (Thread.interrupted()) return;

		logService.info("Load TensorFlow..");
		tensorFlowService.loadLibrary();
		logService.info(tensorFlowService.getStatus().getInfo());

		addCallbackOnEpochDone(new ReduceLearningRateOnPlateau()::reduceLearningRateOnPlateau);
		addCallbackOnCancel(input()::cancel);

	}

	private void initDialog(DenoiSegConfig config) {
		dialog = TrainingProgress.create(this, config.getNumEpochs(), config.getStepsPerEpoch(), statusService, new DefaultThreadService());
		dialog.setTitle("DenoiSeg");
		dialog.setWaitingIcon(getClass().getClassLoader().getResource("bird.gif"), 1, 1, 0, -10);
		inputHandler.setDialog(dialog);
		dialog.addTask("Preparation");
		dialog.addTask("Training");
		dialog.display();
		dialog.setTaskStart(0);
		dialog.setCurrentTaskMessage("Loading TensorFlow");

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
		} catch (CancellationException e) {
			if (stopTraining) return;
			cancel();
			logService.warn("DenoiSeg training canceled.");
		} catch (InterruptedException | ExecutionException e) {
			if (!headless()) uiService.showDialog("Training failed. See console for more details.");
			e.printStackTrace();
			cancel();
		}
	}

	private void mainThread() {
		outputHandler = new DenoiSegOutputHandler(config, this, context);
		addCallbackOnEpochDone(training -> outputHandler.copyBestModel());

		logTrainingStep("Create session..");
		if (Thread.interrupted() || isCanceled()) return;

		try (Graph graph = new Graph();
		     Session sess = new Session(graph)) {

			this.session = sess;

			loadGraph(graph);
			output().initTensors(sess);
			input().finalizeTrainingData();
			if (input().getTrainingData().size() == 0) {
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
			tensorWeightsSegment = makeWeightsTensor();
			tensorWeightsDenoise = makeWeightsTensor();

			if (handleInterruptionOrCancelation()) return;
			logTrainingStep("Start training..");
			if (!headless()) {
				threadService.queue(() -> {
					dialog.setTaskDone(0);
					dialog.setTaskStart(1);
				});
			}
			initPreviewHandler();

			RemainingTimeEstimator timeEstimator = initTimeEstimator();

			for (int epoch = 0; epoch < config().getNumEpochs() && !stopTraining; epoch++) {
				updateTimeEstimator(timeEstimator, epoch);
				runEpoch(training_data, epoch);
				if (handleInterruptionOrCancelation()) return;
			}

//			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			if (!headless()) {
				threadService.queue(() -> dialog.setTaskDone(1));
			}
			stopTraining = true;
			logService.info("Training done.");

//			if (inputs.size() > 0) uiService.show("inputs", Views.stack(inputs));
//			if (targets.size() > 0) uiService.show("targets", Views.stack(targets));

		} catch (IllegalStateException e) {
			cancel();
			e.printStackTrace();
			if (e.getMessage().contains("OOM")) {
				logService.error("Not enough memory available. Try to reduce the training batch size.");
			} else {
				e.printStackTrace();
			}
		} finally {
			if (tensorWeightsSegment != null) tensorWeightsSegment.close();
			if (tensorWeightsDenoise != null) tensorWeightsDenoise.close();
			if (validationTensorData != null) {
				for (Pair<Tensor, Pair<Tensor, Tensor>> pair : validationTensorData) {
					pair.getA().close();
					pair.getB().getA().close();
					pair.getB().getB().close();
				}
			}
		}
	}

	private boolean confirmInputData() {
		InputConfirmationHandler inputConfirmationHandler = new InputConfirmationHandler(context, input());
		boolean confirmed = inputConfirmationHandler.confirmTrainingData();
		if (!confirmed) return false;
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
		if (!headless()) {
			dialog.enableModelSaving();
		}
		if (handleInterruptionOrCancelation()) return;
		training_data.on_epoch_end();
		float validationLoss = validate();
		if (handleInterruptionOrCancelation()) return;
		output().saveCheckpoint(session, previewHandler.getExampleInput(), previewHandler.getExampleOutputDenoise());
		output().setCurrentValidationLoss(validationLoss);
		if (!headless()) {
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
			tensorWeightsDenoise.close();
			tensorWeightsSegment.close();
			return true;
		}
		return false;
	}

	private void initPreviewHandler() {
		previewHandler = new PreviewHandler(context, config().getTrainDimensions());
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
			if (!continueTraining) {
				logService.info("Import graph..");
				output().loadUntrainedGraph(graph);
				output().createSavedModelDirs();
			} else {
				logService.info("Import trained graph..");
				File trainedModel = output().loadTrainedGraph(graph, zipFile);
				output().createSavedModelDirsFromExisting(trainedModel);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void runEpochStep(Session sess, int i, int j, DenoiSegDataWrapper<FloatType> training_data, List<Double> losses) {
		resetBatchIndexIfNeeded();
		ProcessedTrainingData<FloatType> item = training_data.getItem(index);
		runTrainingOp(sess, item);
		if(!isCanceled() && !isStopped()) {
			losses.add((double) output().getCurrentLoss());
			logStatusInConsole(j + 1, config().getStepsPerEpoch());
			if (!headless()) {
				threadService.queue(() -> dialog.updateTrainingProgress(i + 1, j + 1));
			}
			stepsFinished = config().getStepsPerEpoch() * i + j + 1;
			index++;
		}
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
		System.out.println("Training data patches: " + n_train);
		System.out.println("Validation data patches: " + n_val);
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

		ProcessedTrainingDataCollection<FloatType> validationDataList = new ProcessedTrainingDataCollection<>();
		for (int i = 0; i < valData.numBatches(); i++) {
			validationDataList.add(valData.getItem(i));
		}
		this.validationData = validationDataList;
		validationTensorData = new ArrayList<>();
		for (ProcessedTrainingData<FloatType> data : validationDataList) {
			Tensor tensorX = TensorFlowConverter.imageToTensor(data.input, getMapping());
			RandomAccessibleInterval<FloatType> denoised = data.outDenoise;
			RandomAccessibleInterval<FloatType> segmented = data.outSegment;
			Tensor tensorYDenoise = TensorFlowConverter.imageToTensor(denoised, getMapping());
			Tensor tensorYSegment = TensorFlowConverter.imageToTensor(segmented, getMapping());
			validationTensorData.add(new ValuePair<>(tensorX, new ValuePair<>(tensorYDenoise, tensorYSegment)));
		}
	}

	private void normalize() {
		FloatType mean = output().getMean();
		FloatType stdDev = output().getStdDev();
		List<RandomAccessibleInterval<FloatType>> x = new ArrayList<>();
		for (TrainingData<FloatType> pair : input().getTrainingData()) {
			x.add(pair.input);
		}
		mean.set(opService.stats().mean(Views.iterable(Views.stack(x))).getRealFloat());
		stdDev.set(opService.stats().stdDev(Views.iterable(Views.stack(x))).getRealFloat());
		logService.info("mean: " + mean.get());
		logService.info("stdDev: " + stdDev.get());

		TrainUtils.normalize(input().getTrainingData(), mean, stdDev);
		TrainUtils.normalize(input().getValidationData(), mean, stdDev);
	}

	private void runTrainingOp(Session sess, ProcessedTrainingData<FloatType> item) {
//		if(previewCount-- > 0) {
//			opService.context().service(UIService.class).show("input", item.input);
//			opService.context().service(UIService.class).show("denoise", item.outDenoise);
//			opService.context().service(UIService.class).show("segmented", item.outSegment);
//		}
		Tensor tensorX = TensorFlowConverter.imageToTensor(item.input, getMapping());
		Tensor tensorYDenoise = TensorFlowConverter.imageToTensor(item.outDenoise, getMapping());
		Tensor tensorYSegment = TensorFlowConverter.imageToTensor(item.outSegment, getMapping());

		Session.Runner runner = sess.runner();

		Tensor<Float> learningRate = Tensors.create(output().getCurrentLearningRate());
		Tensor<Boolean> learningPhase = Tensors.create(true);
		runner.feed(trainingFeedXOp, tensorX)
				.feed(trainingFeedYDenoiseOp, tensorYDenoise)
				.feed(trainingFeedYSegmentOp, tensorYSegment)
				.feed(trainingFeedLearningPhaseOp, learningPhase)
				.feed(lrAssignOpName, learningRate)
				.feed(trainingFeedSampleWeightsDenoiseOp, tensorWeightsDenoise)
				.feed(trainingFeedSampleWeightsSegmentOp, tensorWeightsSegment)
				.addTarget(trainingTargetOp);
		runner.fetch(trainingFetchLossOp);
		runner.fetch(trainingFetchDenoiseLossOp);
		runner.fetch(trainingFetchSegLossOp);
		runner.fetch(trainingFetchLearningOp);

		List<Tensor<?>> fetchedTensors = runner.run();
		float loss = fetchedTensors.get(0).floatValue();
		float denoiseLoss = fetchedTensors.get(1).floatValue();
		float segLoss = fetchedTensors.get(2).floatValue();
		float newLearningRate = fetchedTensors.get(3).floatValue();

		output().setCurrentLoss(loss);
		output().setCurrentDenoiseLoss(denoiseLoss);
		output().setCurrentSegLoss(segLoss);
		output().setCurrentLearningRate(newLearningRate);

		fetchedTensors.forEach(Tensor::close);
		tensorX.close();
		tensorYDenoise.close();
		tensorYSegment.close();
		learningPhase.close();
		learningRate.close();
	}

	private RandomAccessibleInterval<FloatType> getChannels(RandomAccessibleInterval<FloatType> img, int channelMin, int channelMax) {
		long[] min = new long[img.numDimensions()];
		long[] max = new long[img.numDimensions()];
		img.max(max);
		min[min.length - 1] = channelMin;
		max[max.length - 1] = channelMax;
		IntervalView<FloatType> interval = Views.interval(img, min, max);
//		if(++count < 10) {
//			uiService.show(String.valueOf(count), interval);
//		}
		return interval;
	}

	public void addCallbackOnEpochDone(TrainingCallback callback) {
		onEpochDoneCallbacks.add(callback);
	}

	private int[] getMapping() {
		if (config().getTrainDimensions() == 2) return new int[]{1, 2, 0, 3};
		if (config().getTrainDimensions() == 3) return new int[]{1, 2, 3, 0, 4};
		return new int[0];
	}

	private boolean batchNumSufficient(int n_train) {
		if (config().getTrainBatchSize() > n_train) {
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
		float avgLoss = 0;

		long validationBatches = validationTensorData.size();
		int i = 0;
		for (; i < validationBatches; i++) {

			Pair<Tensor, Pair<Tensor, Tensor>> tensorItem = validationTensorData.get(i);
			ProcessedTrainingData<FloatType> item = validationData.get(i);

			Tensor tensorX = tensorItem.getA();
			Tensor tensorYDenoise = tensorItem.getB().getA();
			Tensor tensorYSegment = tensorItem.getB().getB();
			Tensor<Boolean> tensorLearningPhase = Tensors.create(false);

			Session.Runner runner = session.runner();

			runner.feed(trainingFeedXOp, tensorX)
					.feed(trainingFeedYDenoiseOp, tensorYDenoise)
					.feed(trainingFeedYSegmentOp, tensorYSegment)
					.feed(trainingFeedLearningPhaseOp, tensorLearningPhase)
					.feed(trainingFeedSampleWeightsSegmentOp, tensorWeightsSegment)
					.feed(trainingFeedSampleWeightsDenoiseOp, tensorWeightsDenoise)
					.addTarget(validationTargetOp);
			runner.fetch(trainingFetchLossOp);
			runner.fetch(trainingFetchDenoiseLossOp);
			runner.fetch(trainingFetchSegLossOp);
			if (i == 0) {
				runner.fetch(predictionTargetDenoiseOp);
				runner.fetch(predictionTargetSegmentOp);
			}

			List<Tensor<?>> fetchedTensors = runner.run();

			avgLoss += fetchedTensors.get(0).floatValue();
			avgDenoiseLoss += fetchedTensors.get(1).floatValue();
			avgSegLoss += fetchedTensors.get(2).floatValue();

			if (i == 0) {
				Tensor outputTensorDenoise = fetchedTensors.get(3);
				Tensor outputTensorSegment = fetchedTensors.get(4);
				RandomAccessibleInterval<FloatType> outputDenoise = TensorFlowConverter.tensorToImage(outputTensorDenoise, getMapping());
				RandomAccessibleInterval<FloatType> outputSegment = TensorFlowConverter.tensorToImage(outputTensorSegment, getMapping());
				previewHandler.updateValidationPreview(item.input, outputDenoise, outputSegment, headless(), outputHandler, isStopped() || isCanceled());
//			updateHistoryImage(output);
			}
			fetchedTensors.forEach(Tensor::close);
			tensorLearningPhase.close();

			if (stopTraining || Thread.interrupted() || isCanceled()) {
				i++;
				break;
			}
		}
		avgDenoiseLoss /= (float) (i);
		avgSegLoss /= (float) (i);
		avgLoss /= (float) (i);

		logService.info("\nValidation loss: " + avgLoss + " denoise loss: " + avgDenoiseLoss + " seg loss: " + avgSegLoss);
		return avgLoss;
	}

	public boolean isStopped() {
		return stopTraining;
	}

	public int getStepsFinished() {
		return stepsFinished;
	}

	public void setLearningRate(float newLR) {
		output().setCurrentLearningRate(newLR);
	}

	private void logStatusInConsole(int step, int stepTotal) {
		int maxBareSize = 10; // 10unit for 100%
		int remainProcent = ((100 * step) / stepTotal) / maxBareSize;
		char defaultChar = '-';
		String icon = "*";
		String bare = new String(new char[maxBareSize]).replace('\0', defaultChar) + "]";
		StringBuilder bareDone = new StringBuilder();
		bareDone.append("[");
		for (int i = 0; i < remainProcent; i++) {
			bareDone.append(icon);
		}
		String bareRemain = bare.substring(remainProcent);
		System.out.printf("%d / %d %s%s - loss: %f seg loss: %f denoise loss: %f lr: %f\n", step, stepTotal, bareDone, bareRemain,
				output().getCurrentLoss(),
//				output().getCurrentDenoiSegLoss(),
				output().getCurrentSegLoss(),
				output().getCurrentDenoiseLoss(),
				output().getCurrentLearningRate());
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

	public DenoiSegOutputHandler output() {
		return outputHandler;
	}

	public void saveModel() {
		try {
			File latestModel = this.output().exportLatestTrainedModel();
			ModelZooArchive latestTrainedModel = modelZooService.io().open(latestModel);
			uiService.show("Export current latest Model", latestTrainedModel);
		} catch (IOException e) {
			logService.error(e);
		}
		logService.info("Saved latest trained model to path: " + outputHandler.getMostRecentModelDir().getAbsolutePath());
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

}
