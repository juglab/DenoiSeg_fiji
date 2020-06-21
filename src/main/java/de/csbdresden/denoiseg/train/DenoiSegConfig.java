package de.csbdresden.denoiseg.train;

public class DenoiSegConfig {
	private int numEpochs = 300;
	private int trainBatchSize = 180;
	private int trainPatchShape = 60;
	private int stepsPerEpoch = 200;
	private int neighborhoodRadius = 5;
	private int trainDimensions = 2;
	private float learningRate = 0.0004f;
	private int networkDepth = 4;

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

	public DenoiSegConfig setPatchShape(final int patchShape) {
		trainPatchShape = patchShape;
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

	public long getTrainPatchShape() {
		return trainPatchShape;
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

	public int getNetworkDepth() {
		return networkDepth;
	}

	public void setNetworkDepth(int networkDepth) {
		this.networkDepth = networkDepth;
	}
}
