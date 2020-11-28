package de.csbdresden.denoiseg.train;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

class ProcessedTrainingData<T extends RealType<T>> {
		public RandomAccessibleInterval<T> input;
		public RandomAccessibleInterval<T> outDenoise;
		public RandomAccessibleInterval<T> outSegment;

		public ProcessedTrainingData(RandomAccessibleInterval<T> input, RandomAccessibleInterval<T> outDenoise, RandomAccessibleInterval<T> outSegment) {
			this.input = input;
			this.outSegment = outSegment;
			this.outDenoise = outDenoise;
		}
	}
