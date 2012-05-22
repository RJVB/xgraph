#ifndef _HARD_DEVICES_H
#define _HARD_DEVICES_H

/*
 * Hardcopy Device Header
 *
 * This file declares the types required for the hardcopy table
 * found in hard_devices.c.
 */

#define MFNAME	40
#define LEGEND  MAXBUFSIZE

typedef enum hard_dev_docu_defn { NONE, NO, YES } hard_dev_docu;

typedef struct hard_dev {
    char *dev_name;		/* Device name                */
    int (*dev_init)();		/* Initialization function    */
    char *dev_spec;		/* Default pipe program       */
    char *dev_file;	/* Default file name          */
    char *dev_printer;	/* Default printer name       */
    double dev_max_height, dev_max_width;		/* Default maximum dimension (cm)    */
    char dev_title_font[2*MFNAME];/* Default name of title font        */
    double dev_title_size;	/* Default size of title font (pnts) */
    char dev_legend_font[2*MFNAME];	/* Default name of legend font         */
    double dev_legend_size;	/* Default size of legend font (pnts)  */
    char dev_label_font[2*MFNAME];	/* Default name of label font         */
    double dev_label_size;	/* Default size of label font (pnts)  */
    char dev_axis_font[2*MFNAME];	/* Default name of axis font         */
    double dev_axis_size;	/* Default size of axis font (pnts)  */
    hard_dev_docu dev_docu;	/* Document predicate                */
} Hard_Devices;

extern int hard_count;
extern struct hard_dev *hard_devices;
extern void hard_init();

enum hard_device_names { PS_DEVICE, SPREADSHEET_DEVICE, CRICKET_DEVICE
#ifdef HPGL_DUMP
	, HPGL_DEVICE
#endif
#ifdef IDRAW_DUMP
	, IDRAW_DEVICE=4
#endif
	, XGRAPH_DEVICE, COMMAND_DEVICE
};

#endif
