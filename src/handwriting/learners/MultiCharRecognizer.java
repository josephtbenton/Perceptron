package handwriting.learners;

import handwriting.core.Drawing;
import handwriting.core.RecognizerAI;
import handwriting.core.SampleData;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * Created by josephbenton on 9/28/15.
 */
public class MultiCharRecognizer implements RecognizerAI {
    MultiLayer net;
    double rate;
    int iterations;

    public MultiCharRecognizer() {
        this.net = new MultiLayer(1600, 30, 8);
        rate = 0.1;
        this.iterations = 1000;
    }

    @Override
    public void train(SampleData data, ArrayBlockingQueue<Double> progress) throws InterruptedException {
        for (int j = 0; j < iterations; j++) {
            progress.put((double)j / (double)iterations);
            for (String label: data.allLabels()) {
                for (int i = 0; i < data.numDrawingsFor(label); ++i) {
                    net.train(drawing2doubles(data.getDrawing(label, i)), label2doubles(label), rate);
                }
                net.updateWeights();
            }
        }
    }

    private double[] drawing2doubles(Drawing drawing) {
        int width = drawing.getWidth();
        int height = drawing.getHeight();
        int cur = 0;
        double[] inputs = new double[width * height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                inputs[cur] = drawing.isSet(x, y) ? 1.0 : 0.0;
                cur++;
            }
        }
        return inputs;
    }

    private double[] label2doubles(String label) {
        double[] target = new double[8];
        switch (label) {
            case "1":
                target[0] = 1;
                break;
            case "2":
                target[1] = 1;
                break;
            case "3":
                target[2] = 1;
                break;
            case "4":
                target[3] = 1;
                break;
            case "5":
                target[4] = 1;
                break;
            case "6":
                target[5] = 1;
                break;
            case "7":
                target[6] = 1;
                break;
            case "8":
                target[7] = 1;
                break;
        }
        return target;
    }

    @Override
    public String classify(Drawing d) {
        double[] output = net.compute(drawing2doubles(d));
        int maxIndex = 0;
        for (int i = 0; i < output.length; i++) {
            if (Math.round(output[i]) == 1) {
                maxIndex = i;
            }
        }
        switch (maxIndex) {
            case 0:
                return "1";
            case 1:
                return "2";
            case 2:
                return "3";
            case 3:
                return "4";
            case 4:
                return "5";
            case 5:
                return "6";
            case 6:
                return "7";
            case 7:
                return "8";
        }
        return "Unknown";
    }

}
