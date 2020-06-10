package de.csbdresden.denoiseg.train;

public class DenoiSegConfig {
	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainBatchDimLength = 180;
	private int trainPatchDimLength = 60;
	private int stepsPerEpoch = 200;
	private int neighborhoodRadius = 5;
	private int trainDimensions = 2;
	private float learningRate = 0.0004f;

	public DenoiSegConfig setStepsPerEpoch(final int steps) {
		stepsPerEpoch = steps;
		return this;
	}

	public DenoiSegConfig setNumEpochs(final int numEpochs) {
		this.numEpochs = numEpochs;
		return this;
	}

	public DenoiSegConfig setBatchSize(final int batchSize) {
		trainBatchSize = batchSize;
		return this;
	}

	public DenoiSegConfig setPatchDimLength(final int patchDimLength) {
		trainPatchDimLength = patchDimLength;
		return this;
	}

	public DenoiSegConfig setBatchDimLength(final int batchDimLength) {
		trainBatchDimLength = batchDimLength;
		return this;
	}

	public DenoiSegConfig setTrainDimensions(int trainDimensions) {
		this.trainDimensions = trainDimensions;
		return this;
	}

	public DenoiSegConfig setNeighborhoodRadius(int radius) {
		this.neighborhoodRadius = radius;
		return this;
	}

	public int getTrainDimensions() {
		return trainDimensions;
	}

	public int getNumEpochs() {
		return numEpochs;
	}

	public int getStepsPerEpoch() {
		return stepsPerEpoch;
	}

	public int getTrainBatchSize() {
		return trainBatchSize;
	}

	public long getTrainBatchDimLength() {
		return trainBatchDimLength;
	}

	public long getTrainPatchDimLength() {
		return trainPatchDimLength;
	}

	public int getNeighborhoodRadius() {
		return neighborhoodRadius;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;
	}
}
