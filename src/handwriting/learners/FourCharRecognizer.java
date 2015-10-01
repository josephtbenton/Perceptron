package handwriting.learners;

import handwriting.core.Drawing;
import handwriting.core.RecognizerAI;
import handwriting.core.SampleData;

import java.util.concurrent.ArrayBlockingQueue;

/**
 * Created by josephbenton on 9/28/15.
 */
public class FourCharRecognizer implements RecognizerAI {
    MultiLayer net;
    double rate;
    int iterations;

    public FourCharRecognizer() {
        this.net = new MultiLayer(1600, 40, 4);
        rate = 0.05;
        this.iterations = 800;
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
        double[] target = new double[4];
        switch (label) {
            case "1":
                target[0] = 1.0;
                break;
            case "2":
                target[1] = 1.0;
                break;
            case "3":
                target[2] = 1.0;
                break;
            case "4":
                target[3] = 1.0;
                break;
        }
        return target;
    }

    @Override
    public String classify(Drawing d) {
        double[] output = net.compute(drawing2doubles(d));
        int index = 0;
        for (int i = 0; i < output.length; i++) {
            if (Math.round(output[i]) == 1){
                index = i;
            }
        }
        switch (index) {
            case 0:
                return "1";
            case 1:
                return "2";
            case 2:
                return "3";
            case 3:
                return "4";
        }
        return "Unknown";
    }

}
