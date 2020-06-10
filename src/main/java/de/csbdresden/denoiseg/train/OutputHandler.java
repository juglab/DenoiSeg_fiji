package de.csbdresden.denoiseg.train;

import de.csbdresden.denoiseg.util.N2VUtils;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.io.FileUtils;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class OutputHandler {
	private final DenoiSegConfig config;
	private final DenoiSegTraining training;
	private FloatType mean = new FloatType();

	private FloatType stdDev = new FloatType();

	private float currentLearningRate = 0.0004f;
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

	public OutputHandler(DenoiSegConfig config, DenoiSegTraining training) {
		this.config = config;
		this.currentLearningRate = config.getLearningRate();
		this.training = training;
	}

	public File exportLatestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		ModelSpecification.writeModelConfigFile(config, this, mostRecentModelDir, training.getStepsFinished());
		return N2VUtils.saveTrainedModel(mostRecentModelDir);
	}

	public File exportBestTrainedModel() throws IOException {
		if(noCheckpointSaved) return null;
		return N2VUtils.saveTrainedModel(bestModelDir);
	}

	void copyBestModel(DenoiSegTraining training) {
		if(bestValidationSegLoss > currentValidationSegLoss) {
			bestValidationSegLoss = currentValidationSegLoss;
			try {
				FileUtils.copyDirectory(mostRecentModelDir, bestModelDir);
				ModelSpecification.writeModelConfigFile(config, this, bestModelDir, training.getStepsFinished());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	void createSavedModelDirs() throws IOException {
		bestModelDir = Files.createTempDirectory("n2v-best-").toFile();
		String checkpointDir = Files.createTempDirectory("n2v-latest-").toAbsolutePath().toString() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());
		mostRecentModelDir = new File(checkpointDir).getParentFile();

		checkpointExists = false;

		String predictionGraphDir = config.getTrainDimensions() == 2 ? "prediction_2d" : "prediction_3d";
		byte[] predictionGraphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + predictionGraphDir + "/saved_model.pb") );
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_model.pb"), predictionGraphDef);
	}

	public void createSavedModelDirsFromExisting(File trainedModel) throws IOException {
		mostRecentModelDir = trainedModel;
		bestModelDir = Files.createTempDirectory("n2v-best-").toFile();
		String checkpointDir = mostRecentModelDir.getAbsolutePath() + File.separator + "variables";
		checkpointPrefix = Tensors.create(Paths.get(checkpointDir, "variables").toString());

		checkpointExists = true;

		byte[] predictionGraphDef = IOUtils.toByteArray( new FileInputStream(new File(trainedModel, "saved_model.pb")));
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "saved_model.pb"), predictionGraphDef);
		FileUtils.writeByteArrayToFile(new File(mostRecentModelDir, "training_graph.pb"), predictionGraphDef);
	}

	void loadUntrainedGraph(Graph graph) throws IOException {
		String graphName = config.getTrainDimensions() == 2 ? "graph_2d.pb" : "graph_3d.pb";
		byte[] graphDef = IOUtils.toByteArray( getClass().getResourceAsStream("/" + graphName) );
		graph.importGraphDef( graphDef );
		graph.operations().forEachRemaining( op -> {
			for ( int i = 0; i < op.numOutputs(); i++ ) {
				Output< Object > opOutput = op.output( i );
				String name = opOutput.op().name();
				System.out.println( name );
			}
		} );
	}

	File loadTrainedGraph(Graph graph, File zipFile) throws IOException {

		File trainedModel = Files.createTempDirectory("n2v-imported-model").toFile();
		N2VUtils.unZipAll(zipFile, trainedModel);

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

	void saveCheckpoint(Session sess) {
		sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		noCheckpointSaved = false;
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
