package de.csbdresden.denoiseg.train;

import io.scif.img.ImgSaver;
import net.imagej.modelzoo.specification.DefaultModelSpecification;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
import org.scijava.Context;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Date;

public class OutputHandler {

	private final DenoiSegConfig config;
	private final DenoiSegTraining training;
	private FloatType mean = new FloatType();

	private FloatType stdDev = new FloatType();

	private float currentLearningRate;
	private float currentLoss = Float.MAX_VALUE;
	private float currentDenoisegLoss = Float.MAX_VALUE;
	private float currentDenoiseLoss = Float.MAX_VALUE;
	private float currentSegLoss = Float.MAX_VALUE;
	private float currentValidationSegLoss = Float.MAX_VALUE;
	private float bestValidationSegLoss = Float.MAX_VALUE;

	private File mostRecentModelDir;
	private File bestModelDir;
	private boolean noCheckpointSaved = true;
	private Tensor< String > checkpointPrefix;
	private boolean checkpointExists;
	private ImgSaver imgSaver;

	public OutputHandler(DenoiSegConfig config, DenoiSegTraining training, Context context) {
		this.config = config;
		this.currentLearningRate = config.getLearningRate();
		this.training = training;
		imgSaver = new ImgSaver(context);
	}

	public File exportLatestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
		spec.setName(new Date().toString() + " last checkpoint");
		spec.writeModelConfigFile(config, this, mostRecentModelDir, training.getStepsFinished());
		return TrainUtils.saveTrainedModel(mostRecentModelDir);
	}

	public File exportBestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		return TrainUtils.saveTrainedModel(bestModelDir);
	}

	void copyBestModel(DenoiSegTraining training) {
		if(bestValidationSegLoss > currentValidationSegLoss) {
			bestValidationSegLoss = currentValidationSegLoss;
			try {
				FileUtils.copyDirectory(mostRecentModelDir, bestModelDir);
				DenoiSegModelSpecification spec = new DenoiSegModelSpecification();
				spec.setName(new Date().toString() + " lowest loss");
				spec.writeModelConfigFile(config, this, bestModelDir, training.getStepsFinished());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	void createSavedModelDirs() throws IOException {
		bestModelDir = Files.createTempDirectory("denoiseg-best-").toFile();
		String checkpointDir = Files.createTempDirectory("denoiseg-latest-").toAbsolutePath().toString() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());
		mostRecentModelDir = new File(checkpointDir).getParentFile();

		checkpointExists = false;

		String predictionGraphDir = config.getTrainDimensions() == 2 ? "denoiseg_prediction_2d" : "n2v_prediction_3d";
		byte[] predictionGraphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + predictionGraphDir + "/saved_model.pb") );
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_model.pb"), predictionGraphDef);
	}

	public void createSavedModelDirsFromExisting(File trainedModel) throws IOException {
		mostRecentModelDir = trainedModel;
		bestModelDir = Files.createTempDirectory("denoiseg-best-").toFile();
		String checkpointDir = mostRecentModelDir.getAbsolutePath() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());

		checkpointExists = true;

		byte[] predictionGraphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "saved_model.pb")));
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_graph.pb"), predictionGraphDef);
	}

	void loadUntrainedGraph(Graph graph) throws IOException {
		String graphName = config.getTrainDimensions() == 2 ? "denoiseg_graph_2d.pb" : "denoiseg_graph_3d.pb";
		byte[] graphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + graphName) );
		graph.importGraphDef( graphDef );
//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				System.out.println( name );
//			}
//		} );
	}

	File loadTrainedGraph(Graph graph, File zipFile) throws IOException {

		File trainedModel = Files.createTempDirectory("denoiseg-imported-model").toFile();
		TrainUtils.unZipAll(zipFile, trainedModel);

		byte[] graphDef = new byte[ 0 ];
		try {
			graphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "training_graph.pb")));
		} catch ( IOException e ) {
			e.printStackTrace();
		}
		graph.importGraphDef( graphDef );

//		graph.operations().forEachRemaining( op -> {
//			for ( int i = 0; i < op.numOutputs(); i++ ) {
//				Output< Object > opOutput = op.output( i );
//				String name = opOutput.op().name();
//				logService.info( name );
//			}
//		} );
		return trainedModel;
	}


	public void initTensors(Session sess) {
		if (checkpointExists) {
			sess.runner()
					.feed("save/Const", checkpointPrefix)
					.addTarget("save/restore_all").run();
		} else {
			sess.runner().addTarget("init").run();
		}
	}

	void saveCheckpoint(Session sess, RandomAccessibleInterval<FloatType> input, RandomAccessibleInterval<FloatType> output) {
		sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		noCheckpointSaved = false;
		if(input != null && output != null) {
			imgSaver.saveImg(new File(mostRecentModelDir, new DefaultModelSpecification().getTestInput()).getAbsolutePath(),
					toImg(input));
			imgSaver.saveImg(new File(mostRecentModelDir, new DefaultModelSpecification().getTestOutput()).getAbsolutePath(),
					toImg(output));
		}
	}

	private Img<?> toImg(RandomAccessibleInterval<FloatType> input) {
		ArrayImg<FloatType, ?> res = new ArrayImgFactory<>(new FloatType()).create(input);
		LoopBuilder.setImages(input, res).forEachPixel((in, out) -> {
			out.set(in);
		});
		return res;
	}

	public float getCurrentSegLoss() {
		return currentSegLoss;
	}

	public float getCurrentDenoiseLoss() {
		return currentDenoiseLoss;
	}

	public float getCurrentLearningRate() {
		return currentLearningRate;
	}

	public void setCurrentLearningRate(float rate) {
		currentLearningRate = rate;
	}

	public FloatType getMean() {
		return mean;
	}

	public FloatType getStdDev() {
		return stdDev;
	}

	public void setCurrentDenoiseLoss(float loss) {
		this.currentDenoiseLoss = loss;
	}

	public void setCurrentSegLoss(float abs) {
		this.currentSegLoss = abs;
	}

	public float getCurrentValidationSegLoss() {
		return currentValidationSegLoss;
	}

	public void setCurrentValidationSegLoss(float loss) {
		this.currentValidationSegLoss = loss;
	}

	public void dispose() {
		if(checkpointPrefix != null) checkpointPrefix.close();
	}

	public void setCurrentDenoisegLoss(float loss) {
		this.currentDenoisegLoss = loss;
	}

	public void setCurrentLoss(float loss) {
		this.currentLoss = loss;
	}

	public float getCurrentLoss() {
		return currentLoss;
	}

	public float getCurrentDenoiSegLoss() {
		return currentDenoisegLoss;
	}
}
