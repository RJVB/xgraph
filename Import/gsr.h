#ifndef _GSR_H

/* Definitions for the '.gsr' format used for neuro-vegetative recordings */

/* The general structure of the file is:
 \ General header: global parameters relating to the recording (GSRHeader)
 \ 	Channel-header: describing a single channel (GSRChannelHeader)
 \		channel data: N* unsigned short (GSRSample), N given in the preceding GSRChannelHeader
 \	<repeat for all channels>
 \ "Event zone":
 \ 	Field describing event e (GSREventRecord)
 \	<repeat for all events and until EOF>
 */

  /* Placeholder conditional just for the case we go to a platform where sizeof(long)!=4 || sizeof(short)!=2 */
#if 1
	typedef unsigned long GSRlong;
	typedef short GSRshort;
#endif

/* GSRHeadersRaw structure: uses only chars (bytes). This is necessary because the programme generating
 \ the .gsr files uses byte-alignment. Note that therefore the padding string at the end is 4 bytes
 \ longer. Under GCC 3 on a PIII, there was 2byte padding between NChannels and TResolution (plus
 \ at some other position I didn't check).
 \ This does mean, of course, that we'll have to copy the fields one by one, taking care of the
 \ proper casting... As we define all the Raw fields as pointers, this is not very difficult:
 \ *( (float*) gheaderRaw.TResolution ) will cast the TResolution field to the float it ought to be.
 */
typedef struct GSRHeadersRaw{
	char magic[8];
	unsigned char NChannels[sizeof(GSRshort)];
	unsigned char TResolution[sizeof(float)];
	unsigned char TResolutionCorrection[sizeof(double)];
	unsigned char Samples[sizeof(GSRlong)];
	unsigned char theChannel[sizeof(GSRshort)];
	unsigned char CursorPosition[sizeof(GSRlong)];
	unsigned char FirstShownSample[sizeof(GSRlong)];
	unsigned char ShownSamples[sizeof(GSRlong)];
	unsigned char FirstLoadedSample[sizeof(GSRlong)];
	  /* According to specs, AbsoluteTime should be a float, but if so, NEvents is not correct. */
	unsigned char AbsoluteTime[sizeof(double)];
	unsigned char NEvents[sizeof(GSRshort)];
	unsigned char GridState[sizeof(GSRshort)];
	unsigned char bgColour[sizeof(GSRshort)];
	unsigned char TracingSpeed[sizeof(GSRshort)];
	char padding[964];
} GSRHeadersRaw;

typedef struct GSRHeaders{
	char magic[8];
	GSRshort NChannels;
	float TResolution;
	double TResolutionCorrection;
	GSRlong Samples;
	GSRshort theChannel;
	GSRlong CursorPosition;
	GSRlong FirstShownSample;
	GSRlong ShownSamples;
	GSRlong FirstLoadedSample;
	  /* ought to be float?! */
	union{
		float fAT[2];
		double dAT;
	} AbsoluteTime;
	GSRshort NEvents;
	GSRshort GridState;
	GSRshort bgColour;
	GSRshort TracingSpeed;
	char padding[964];
} GSRHeaders;

/* Idem, the channel header needs to be byte-aligned. This resuls
 \ in a similar (4byte) size difference. HOWEVER, for whatever reason,
 \ we need to NOT take this into account!!
 */
typedef struct GSRChannelHeadersRaw{
	char ChannelName[2];
	unsigned char ChannelCoefficient[sizeof(GSRshort)];
	unsigned char ChannelOffset[sizeof(GSRshort)];
	unsigned char SelectedGain[sizeof(float)];
	unsigned char SelectedOffset[sizeof(float)];
	unsigned char ChannelState[sizeof(GSRshort)];
	unsigned char ChannelColour[sizeof(GSRshort)];
	unsigned char ChannelUnits[8][sizeof(char)];
	  /* padding vector that is 4 bytes short than it should be according to the docs. */
	char padding[230];
} GSRChannelHeadersRaw;

typedef struct GSRChannelHeaders{
	char ChannelName[2];
	GSRshort ChannelCoefficient;
	GSRshort ChannelOffset;
	float SelectedGain;
	float SelectedOffset;
	GSRshort ChannelState;
	GSRshort ChannelColour;
	char ChannelUnits[8];
	  /* padding vector with the documented length. No real interest, as this structure is not for direct import. */
	char padding[230];
} GSRChannelHeaders;

typedef GSRshort GSRSamples;

typedef struct GSREvenRecords{
	GSRlong EventPosition,
		EventColour;
	char EventText[80];
} GSREventRecords;

#define _GSR_H
#endif
