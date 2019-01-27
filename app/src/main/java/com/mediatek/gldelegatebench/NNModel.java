package com.mediatek.gldelegatebench;

class NNModel {
    private String modelName;
    private int[] inputShapes;
    private int[] outputShapes;

    public NNModel(String name, int[] iShape, int[] oShape) {
        modelName = name;
        inputShapes = iShape;
        outputShapes = oShape;
    }
    public String getModelName() {
        return modelName;
    }

    public int[] getInputShapes() {
        return inputShapes;
    }

    public int[] getOutputShapes() {
        return outputShapes;
    }
}
