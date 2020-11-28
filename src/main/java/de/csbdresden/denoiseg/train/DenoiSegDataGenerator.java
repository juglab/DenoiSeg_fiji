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

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.scijava.log.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DenoiSegDataGenerator {

	static <T extends RealType<T> & NativeType<T>> void augment(TrainingDataCollection<T> data) {
		if(data.get(0).input.dimension(0) == data.get(0).input.dimension(1)) {
			//share in XY
			augmentBatches(data);
		}
//		Collections.shuffle(patches);
//		for (int i = 0; i < x.size(); i++) {
//			RandomAccessibleInterval<T> patch = x.get(i);
//			RandomAccessibleInterval<T> rai = Views.addDimension(patch, 0, 0);
//			rai = Views.addDimension(rai, 0, 0);
//			tiles.set(i, N2VUtils.copy(Views.zeroMin(rai)));
//			tiles.set(i, Views.zeroMin(rai));
//		}
	}

	private static <T extends RealType<T>> TrainingDataCollection<T> extractBatches(
			RandomAccessibleInterval<T> img,
			RandomAccessibleInterval<T> labeling,
			Interval shape) {
		if(img.numDimensions() == shape.numDimensions()) return extractBatchesNoSlicing(img, labeling, shape);
		TrainingDataCollection<T> res = new TrainingDataCollection<>();
		for (int i = 0; i < img.dimension(shape.numDimensions()); i++) {
			IntervalView<T> img1 = Views.hyperSlice(img, shape.numDimensions(), i);
			IntervalView<T> labeling1 = Views.hyperSlice(labeling, shape.numDimensions(), i);
			if(img1.numDimensions() == shape.numDimensions()) {
				res.addAll(extractBatchesNoSlicing(img1, labeling1, shape));
			} else {
				for (int j = 0; j < img1.dimension(shape.numDimensions()); j++) {
					IntervalView<T> img2 = Views.hyperSlice(img1, shape.numDimensions(), j);
					IntervalView<T> labeling2 = Views.hyperSlice(labeling1, shape.numDimensions(), j);
					if(img2.numDimensions() == shape.numDimensions()) {
						res.addAll(extractBatchesNoSlicing(img2, labeling2, shape));
					} else {
						for (int k = 0; k < img2.dimension(shape.numDimensions()); k++) {
							IntervalView<T> img3 = Views.hyperSlice(img2, shape.numDimensions(), k);
							IntervalView<T> labeling3 = Views.hyperSlice(labeling2, shape.numDimensions(), k);
							res.addAll(extractBatchesNoSlicing(img3, labeling3, shape));

						}
					}
				}
			}
		}
		return res;
	}

	private static <T extends RealType<T>> List<RandomAccessibleInterval<T>> extractBatches(
			RandomAccessibleInterval<T> img,
			Interval shape) {
		if(img.numDimensions() == shape.numDimensions()) return extractBatchesNoSlicing(img, shape);
		List<RandomAccessibleInterval<T>> res = new ArrayList<>();
		for (int i = 0; i < img.dimension(shape.numDimensions()); i++) {
			IntervalView<T> img1 = Views.hyperSlice(img, shape.numDimensions(), i);
			if(img1.numDimensions() == shape.numDimensions()) {
				res.addAll(extractBatchesNoSlicing(img1, shape));
			} else {
				for (int j = 0; j < img1.dimension(shape.numDimensions()); j++) {
					IntervalView<T> img2 = Views.hyperSlice(img1, shape.numDimensions(), j);
					if(img2.numDimensions() == shape.numDimensions()) {
						res.addAll(extractBatchesNoSlicing(img2, shape));
					} else {
						for (int k = 0; k < img2.dimension(shape.numDimensions()); k++) {
							IntervalView<T> img3 = Views.hyperSlice(img2, shape.numDimensions(), k);
							res.addAll(extractBatchesNoSlicing(img3, shape));

						}
					}
				}
			}
		}
		return res;
	}

	private static <T extends RealType<T>> TrainingDataCollection<T> extractBatchesNoSlicing(
			RandomAccessibleInterval<T> img,
			RandomAccessibleInterval<T> labeling,
			Interval shape) {
		TrainingDataCollection<T> res = new TrainingDataCollection<>();
		if(shapeTooBig(img, shape)) {
			System.out.println("DenoiSegDataGenerator::extractPatchesNoSlicing: 'shape' is too big");
			return res;
		}
		if(shape.numDimensions() == 2) extractBatches2D(img, labeling, shape, res);
		else if(shape.numDimensions() == 3) extractBatches3D(img, labeling, shape, res);
		return res;
	}

	private static <T extends RealType<T>> List<RandomAccessibleInterval<T>> extractBatchesNoSlicing(
			RandomAccessibleInterval<T> img,
			Interval shape) {
		List<RandomAccessibleInterval<T>> res = new ArrayList<>();
		if(shapeTooBig(img, shape)) {
			System.out.println("DenoiSegDataGenerator::extractPatchesNoSlicing: 'shape' is too big");
			return res;
		}
		if(shape.numDimensions() == 2) extractBatches2D(img, shape, res);
		else if(shape.numDimensions() == 3) extractBatches3D(img, shape, res);
		return res;
	}

	private static <T extends RealType<T>> void extractBatches2D(
			RandomAccessibleInterval<T> img,
			RandomAccessibleInterval<T> labeling,
			Interval shape,
			TrainingDataCollection<T> res) {
		for (int y = 0; y <= img.dimension(1) - shape.dimension(1); y+=shape.dimension(1)) {
			for (int x = 0; x <= img.dimension(0) - shape.dimension(0); x+=shape.dimension(0)) {
				long[] min = {x, y};
				long[] minLabeling = {min[0], min[1], 0};
				long[] max = {x + shape.dimension(0)-1, y + shape.dimension(1)-1};
				long[] maxLabeling = {max[0], max[1], 2};
//					System.out.println(res.size() + ": " + Arrays.toString(min) + " -> " + Arrays.toString(max));
				IntervalView<T> n2vTile = Views.interval(img,
						min,
						max);
				IntervalView<T> labelingTile = Views.interval(labeling,
						minLabeling,
						maxLabeling);
				res.add(new TrainingData<>(Views.zeroMin(n2vTile), Views.zeroMin(labelingTile)));
			}
		}
	}

	private static <T extends RealType<T>> void extractBatches2D(
			RandomAccessibleInterval<T> img,
			Interval shape,
			List<RandomAccessibleInterval<T>> res) {
		for (int y = 0; y <= img.dimension(1) - shape.dimension(1); y+=shape.dimension(1)) {
			for (int x = 0; x <= img.dimension(0) - shape.dimension(0); x+=shape.dimension(0)) {
				long[] min = {x, y};
				long[] max = {x + shape.dimension(0)-1, y + shape.dimension(1)-1};
				long[] maxLabeling = {max[0], max[1], 2};
//					System.out.println(res.size() + ": " + Arrays.toString(min) + " -> " + Arrays.toString(max));
				IntervalView<T> n2vTile = Views.interval(img,
						min,
						max);
				res.add(Views.zeroMin(n2vTile));
			}
		}
	}

	private static <T extends RealType<T>> void extractBatches3D(
			RandomAccessibleInterval<T> img,
			RandomAccessibleInterval<T> labeling,
			Interval shape,
			TrainingDataCollection<T> res) {
		for (int z = 0; z <= img.dimension(2) - shape.dimension(2); z+=shape.dimension(2)) {
			for (int y = 0; y <= img.dimension(1) - shape.dimension(1); y += shape.dimension(1)) {
				for (int x = 0; x <= img.dimension(0) - shape.dimension(0); x += shape.dimension(0)) {
					long[] min = {x, y, z};
					long[] minLabeling = {min[0], min[1], min[2], 0};
					long[] max = {x + shape.dimension(0) - 1, y + shape.dimension(1) - 1, z + shape.dimension(2) - 1};
					long[] maxLabeling = {max[0], max[1], max[2], 2};
//					System.out.println(res.size() + ": " + Arrays.toString(min) + " -> " + Arrays.toString(max));
					IntervalView<T> n2vTile = Views.interval(img,
							min,
							max);
					IntervalView<T> labelingTile = Views.interval(labeling,
							minLabeling,
							maxLabeling);
					res.add(new TrainingData<>(Views.zeroMin(n2vTile), Views.zeroMin(labelingTile)));
				}
			}
		}
	}

	private static <T extends RealType<T>> void extractBatches3D(
			RandomAccessibleInterval<T> img,
			Interval shape,
			List<RandomAccessibleInterval<T>> res) {
		for (int z = 0; z <= img.dimension(2) - shape.dimension(2); z+=shape.dimension(2)) {
			for (int y = 0; y <= img.dimension(1) - shape.dimension(1); y += shape.dimension(1)) {
				for (int x = 0; x <= img.dimension(0) - shape.dimension(0); x += shape.dimension(0)) {
					long[] min = {x, y, z};
					long[] max = {x + shape.dimension(0) - 1, y + shape.dimension(1) - 1, z + shape.dimension(2) - 1};
//					System.out.println(res.size() + ": " + Arrays.toString(min) + " -> " + Arrays.toString(max));
					IntervalView<T> n2vTile = Views.interval(img,
							min,
							max);
					res.add(Views.zeroMin(n2vTile));
				}
			}
		}
	}

	private static <T extends RealType<T>> boolean shapeTooBig(RandomAccessibleInterval<T> img, Interval shape) {
		for (int i = 0; i < shape.numDimensions(); i++) {
			if(shape.dimension(i) > img.dimension(i)) return true;
		}
		return false;
	}

	private static <T extends RealType<T>> void augmentBatches(TrainingDataCollection<T> patches) {
		TrainingDataCollection<T> augmented = new TrainingDataCollection<>();
		patches.forEach(patch -> {
			IntervalView<T> r1A = Views.zeroMin(Views.rotate(patch.input, 0, 1));
			IntervalView<T> r1B = Views.zeroMin(Views.rotate(patch.outSegment, 0, 1));
			IntervalView<T> r2A = Views.zeroMin(Views.rotate(r1A, 0, 1));
			IntervalView<T> r2B = Views.zeroMin(Views.rotate(r1B, 0, 1));
			augmented.add(new TrainingData<>(r1A, r1B));
			augmented.add(new TrainingData<>(r2A, r2B));
			IntervalView<T> r3A = Views.zeroMin(Views.rotate(r2A, 0, 1));
			IntervalView<T> r3B = Views.zeroMin(Views.rotate(r2B, 0, 1));
			augmented.add(new TrainingData<>(r3A, r3B));
		});
		patches.addAll(augmented);
		augmented.clear();
		for (TrainingData<T> patch : patches) {
			IntervalView<T> iA = Views.zeroMin(Views.invertAxis(patch.input, 0));
			IntervalView<T> iB = Views.zeroMin(Views.invertAxis(patch.outSegment, 0));
			augmented.add(new TrainingData<>(iA, iB));
		}
		patches.addAll(augmented);
	}

	static TrainingDataCollection<FloatType> createTiles(
			RandomAccessibleInterval< FloatType > inputRAI,
			RandomAccessibleInterval<FloatType> labelingRAI,
			int trainDimensions, long patchShape, Logger logger) {

		long superPatchShape = getSmallestInputDim(inputRAI, trainDimensions);
		superPatchShape = Math.min(superPatchShape, patchShape*2);

		long[] batchShapeData = new long[trainDimensions];
		Arrays.fill(batchShapeData, superPatchShape);
		FinalInterval batchShape = new FinalInterval(batchShapeData);
//		logger.info( "Creating tiles of size " + Arrays.toString(Intervals.dimensionsAsIntArray(batchShape)) + ".." );
		//		long[] tiledim = new long[ tiles.get( 0 ).getA().numDimensions() ];
//		tiles.get( 0 ).getA().dimensions( tiledim );
//		logger.info( "Generated " + tiles.size() + " tiles of shape " + Arrays.toString( tiledim ) );
//		RandomAccessibleInterval<FloatType> tilesStack = Views.stack(tiles);
//		uiService.show("tiles", tilesStack);
		return extractBatches(inputRAI, labelingRAI, batchShape);
	}

	static List<RandomAccessibleInterval<FloatType>> createTiles(
			RandomAccessibleInterval< FloatType > inputRAI,
			int trainDimensions, long patchShape, Logger logger ) {

		long superPatchShape = getSmallestInputDim(inputRAI, trainDimensions);
		superPatchShape = Math.min(superPatchShape, patchShape*2);
		long[] batchShapeData = new long[trainDimensions];
		Arrays.fill(batchShapeData, superPatchShape);
		FinalInterval batchShape = new FinalInterval(batchShapeData);
//		logger.info( "Creating tiles of size " + Arrays.toString(Intervals.dimensionsAsIntArray(batchShape)) + ".." );
		//		long[] tiledim = new long[ tiles.get( 0 ).numDimensions() ];
//		tiles.get( 0 ).dimensions( tiledim );
//		logger.info( "Generated " + tiles.size() + " tiles of shape " + Arrays.toString( tiledim ) );
//		RandomAccessibleInterval<FloatType> tilesStack = Views.stack(tiles);
//		uiService.show("tiles", tilesStack);
		return extractBatches(inputRAI, batchShape);
	}

	private static long getSmallestInputDim(RandomAccessibleInterval<FloatType> img, int maxDimensions) {
		long res = img.dimension(0);
		for (int i = 1; i < img.numDimensions() && i < maxDimensions; i++) {
			if(img.dimension(i) < res) res = img.dimension(i);
		}
		return res;
	}
}
