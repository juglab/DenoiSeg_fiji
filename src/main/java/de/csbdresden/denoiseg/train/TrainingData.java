package de.csbdresden.denoiseg.train;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

public class TrainingData<T extends RealType<T>> {
	public RandomAccessibleInterval<T> input;
	public RandomAccessibleInterval<T> outSegment;

	public TrainingData(RandomAccessibleInterval<T> input, RandomAccessibleInterval<T> outSegment) {
		this.input = input;
		this.outSegment = outSegment;
	}
}
