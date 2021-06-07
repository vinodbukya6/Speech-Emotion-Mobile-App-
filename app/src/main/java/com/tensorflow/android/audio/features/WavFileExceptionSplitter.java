package com.tensorflow.android.audio.features;

@SuppressWarnings("serial")
public class WavFileExceptionSplitter extends Exception
{
	public WavFileExceptionSplitter()
	{
		super();
	}

	public WavFileExceptionSplitter(String message)
	{
		super(message);
	}

	public WavFileExceptionSplitter(String message, Throwable cause)
	{
		super(message, cause);
	}

	public WavFileExceptionSplitter(Throwable cause) 
	{
		super(cause);
	}
}
