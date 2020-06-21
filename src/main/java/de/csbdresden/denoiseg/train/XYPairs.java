package de.csbdresden.denoiseg.train;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Pair;

import java.util.ArrayList;

public class XYPairs<T extends RealType<T>> extends ArrayList<Pair<RandomAccessibleInterval<T>, RandomAccessibleInterval<T>>> { }
