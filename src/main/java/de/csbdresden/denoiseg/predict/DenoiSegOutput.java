package de.csbdresden.denoiseg.predict;

import net.imagej.modelzoo.consumer.model.prediction.PredictionOutput;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class DenoiSegOutput<T extends RealType<T> & NativeType<T>, B extends RealType<B> & NativeType<B>> implements PredictionOutput {

	private RandomAccessibleInterval<T> denoised;
	private RandomAccessibleInterval<B> segmented;

	public RandomAccessibleInterval<T> getDenoised() {
		return denoised;
	}

	public RandomAccessibleInterval<B> getSegmented() {
		return segmented;
	}

	public DenoiSegOutput(RandomAccessibleInterval<T> denoised, RandomAccessibleInterval<B> segmented) {
		this.denoised = denoised;
		this.segmented = segmented;
	}

	@Override
	public Map<String, Object> asMap() {
		Map<String, Object> map = new HashMap<>();
		map.put("denoised", denoised);
		map.put("segmented", segmented);
		return Collections.unmodifiableMap(map);
	}
}
