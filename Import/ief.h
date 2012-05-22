#ifndef _IEF_H

/* Definitions for the '.IEF' format used for instrumented-vehicle recordings */

/* The general structure of the file is very simple: a single "line" per datafield, consisting of:
 \ int32 time; int8 topsFW, topsRW; int2 telecommande, brake; int12 steer;
 \ int16 refX, lX, aX; # ref.value for X sensor; X acceleration, X angular acceleration
 \ int16 refY, lY, aY;
 \ int16 refZ, lZ, aZ;
 \ int16 throttle;
 \ int32 time2; # for verification
 */

  /* Placeholder conditional just for the case we go to a platform where sizeof(long)!=4 || sizeof(short)!=2 */
#include "64typedefs.h"

#ifdef __GNUC__
#	define GCC_PACKED	__attribute__ ((packed))
#else
#	define GCC_PACKED	/**/
#endif

#undef USING_BITFIELDS

// nguidon | chN_guidon | Cligno gauche | frein | Cligno droit | NC | 11bits position du guidon
// NC should be removed, or one of nguidon/chN_guidon because brake (frein) is at bit 13...

#ifdef USING_BITFIELDS
#	if defined(i386) || defined(__LITTLE_ENDIAN__) || (defined(BYTE_ORDER) && BYTE_ORDER==LITTLE_ENDIAN)
#warning "bitfields on a Little endian host"
	/* definitions for byte-reversed-ordered machines:	*/
		typedef struct IEFcommands {
			int16 steer:11;
			uint8 unused:1, turnRight:1, brake:1, turnLeft:1, chN_steer:1;
		} GCC_PACKED IEFcommands;
		typedef struct IEFcommands2 {
			int16 steer:11;
			uint8 TTLMask:5;
		} GCC_PACKED IEFcommands2;
#	else
		typedef struct IEFcommands {
			 // 20100525: possibly turn left& right not interchanged w.r.t. little endian order?!
			uint8 chN_steer:1, turnLeft:1, brake:1, turnRight:1, unused:1;
			int16 steer:11;
		} GCC_PACKED IEFcommands;
		typedef struct IEFcommands2 {
			uint8 TTLMask:5;
			int16 steer:11;
		} GCC_PACKED IEFcommands2;
#	endif

	typedef union IEFbinaries {
		IEFcommands comm;
		uint16 dword;
	} GCC_PACKED IEFbinaries;
#else
	typedef struct IEFcommands {
		uint8 TTLMask, brake, remote;
		int16 steer;
	} GCC_PACKED IEFcommands;
	typedef IEFcommands	IEFcommands2;
#endif

typedef struct IEFsamples{
	uint32 time;
	uint8 topsRW, topsFW;
	  // handlebar (steer) and a number of binary switchgear:
	uint16 comm;
	struct{
		uint16 X, Y, Z;
	} ref, lin, ang;
	int16 throttle;
	uint32 time2;
} GCC_PACKED IEFsamples;


#define IEF_CHANNELS	17

typedef struct IEFsensorSpecs{
	uint16 bitResolution, channels, offset;
	double reading2ISO;
} IEFsensorSpecs;

typedef struct IEFmobikeSpecs{
	double wheelbase, rake, steerSensorGearing;
	uint16 steerSensorResolution;
} IEFmobikeSpecs;

typedef enum { c_steer, c_fw, c_rw, c_refx, c_refy, c_refz, c_ax, c_ay, c_az, c_rx, c_ry, c_rz, c_throttle, c_brake } IEFmobikeSensors;

#define _IEF_H
#endif
