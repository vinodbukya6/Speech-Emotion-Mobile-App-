package com.tensorflow.android.audio.features;

import java.io.*;
import java.net.URL;
import java.util.Arrays;
import java.lang.*;
import android.os.Environment;


/**
 * Splits WAV-files in multiple parts.
 * This class splits a big WAV-file in multiple WAV-file, each with a fixed length (SPLIT_FILE_LENGTH_MS).
 * It takes it input file from an embedded resource, and writes a series of out*.wav files.
 *
 * @author Jeroen De Swaef
 */
@SuppressWarnings("unused")
public class WaveSplitter {
    public static void main(String[] args) throws Exception {
        String sample = null;
        File audioFilePath1 = new File(sample);
        File[] rowdata = getChunks(audioFilePath1);

    }

    public static File[] getChunks (File audioFilePath ) throws Exception
    {
            int SPLIT_FILE_LENGTH_MS = 10000;
            File outputChunks[] =  new File[1000];

            //WavFileSplitter readWavFile = openWavFile(new File(filename));
            WavFileSplitter inputWavFile = WavFileSplitter.openWavFile(audioFilePath);
            
            
            // Get the number of audio channels in the wav file
            int numChannels = inputWavFile.getNumChannels();
            // set the maximum number of frames for a target file,
            // based on the number of milliseconds assigned for each file
            int maxFramesPerFile = (int) inputWavFile.getSampleRate() * SPLIT_FILE_LENGTH_MS / 1000;

            // Create a buffer of maxFramesPerFile frames
            double[] buffer = new double[maxFramesPerFile * numChannels];

            int framesRead;
            int fileCount = 1;
            do {
                // Read frames into buffer
                framesRead = inputWavFile.readFrames(buffer, maxFramesPerFile);
                String baseDir = Environment.getExternalStorageDirectory().getAbsolutePath();
                // path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
                //File file = new File(path, "/" + fname);
                WavFileSplitter outputWavFile = WavFileSplitter.newWavFile(
                        new File(baseDir, "/" + "audioData"+ "/" + "chunk" + (fileCount) + ".wav"),
                        inputWavFile.getNumChannels(),
                        framesRead,
                        inputWavFile.getValidBits(),
                        inputWavFile.getSampleRate());

                // Write the buffer
                outputWavFile.writeFrames(buffer, framesRead);
                outputWavFile.close();
                // System.out.printf("%d %d\n", framesRead, outputWavFile);
                File out = new File(String.valueOf(outputWavFile));
                outputChunks[fileCount] = out;
                fileCount++;
            } while (framesRead != 0);{
              inputWavFile.close();
              //outputWavFile.close();
    }

        return outputChunks;
    }
}