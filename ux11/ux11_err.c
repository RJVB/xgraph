/*
 * X11 Error Message Formatter
 */

#include <stdio.h>
#include "ux11.h"
#include "ux11_internal.h"
#include <X11/Xproto.h>

#define MAX_ERR_LEN	1024

char *event_name( type)
int type;
{
	switch( type){
		case KeyPress:
			return( "KeyPress" );
		case KeyRelease:
			return( "KeyRelease" );
		case ButtonPress:
			return( "ButtonPress" );
		case ButtonRelease:
			return( "ButtonRelease" );
		case MotionNotify:
			return( "MotionNotify" );
		case EnterNotify:
			return( "EnterNotify" );
		case LeaveNotify:
			return( "LeaveNotify" );
		case FocusIn:
			return( "FocusIn" );
		case FocusOut:
			return( "FocusOut" );
		case KeymapNotify:
			return( "KeymapNotify" );
		case Expose:
			return( "Expose" );
		case GraphicsExpose:
			return( "GraphicsExpose" );
		case NoExpose:
			return( "NoExpose" );
		case VisibilityNotify:
			return( "VisibilityNotify" );
		case CreateNotify:
			return( "CreateNotify" );
		case DestroyNotify:
			return( "DestroyNotify" );
		case UnmapNotify:
			return( "UnmapNotify" );
		case MapNotify:
			return( "MapNotify" );
		case MapRequest:
			return( "MapRequest" );
		case ReparentNotify:
			return( "ReparentNotify" );
		case ConfigureNotify:
			return( "ConfigureNotify" );
		case ConfigureRequest:
			return( "ConfigureRequest" );
		case GravityNotify:
			return( "GravityNotify" );
		case ResizeRequest:
			return( "ResizeRequest" );
		case CirculateNotify:
			return( "CirculateNotify" );
		case CirculateRequest:
			return( "CirculateRequest" );
		case PropertyNotify:
			return( "PropertyNotify" );
		case SelectionClear:
			return( "SelectionClear" );
		case SelectionRequest:
			return( "SelectionRequest" );
		case SelectionNotify:
			return( "SelectionNotify" );
		case ColormapNotify:
			return( "ColormapNotify" );
		case ClientMessage:
			return( "ClientMessage" );
		case MappingNotify:
			return( "MappingNotify" );
		default:
			return( "Unknown event type");
		}
}

char *req_str(req_code)
int req_code;
/*
 * Returns a simple text representation for a request code
 */
{
    switch (req_code) {
		case X_CreateWindow:
			return "X_CreateWindow";
		case X_ChangeWindowAttributes:
			return "X_ChangeWindowAttributes";
		case X_GetWindowAttributes:
			return "X_GetWindowAttributes";
		case X_DestroyWindow:
			return "X_DestroyWindow";
		case X_DestroySubwindows:
			return "X_DestroySubwindows";
		case X_ChangeSaveSet:
			return "X_ChangeSaveSet";
		case X_ReparentWindow:
			return "X_ReparentWindow";
		case X_MapWindow:
			return "X_MapWindow";
		case X_MapSubwindows:
			return "X_MapSubwindows";
		case X_UnmapWindow:
			return "X_UnmapWindow";
		case X_UnmapSubwindows:
			return "X_UnmapSubwindows";
		case X_ConfigureWindow:
			return "X_ConfigureWindow";
		case X_CirculateWindow:
			return "X_CirculateWindow";
		case X_GetGeometry:
			return "X_GetGeometry";
		case X_QueryTree:
			return "X_QueryTree";
		case X_InternAtom:
			return "X_InternAtom";
		case X_GetAtomName:
			return "X_GetAtomName";
		case X_ChangeProperty:
			return "X_ChangeProperty";
		case X_DeleteProperty:
			return "X_DeleteProperty";
		case X_GetProperty:
			return "X_GetProperty";
		case X_ListProperties:
			return "X_ListProperties";
		case X_SetSelectionOwner:
			return "X_SetSelectionOwner";
		case X_GetSelectionOwner:
			return "X_GetSelectionOwner";
		case X_ConvertSelection:
			return "X_ConvertSelection";
		case X_SendEvent:
			return "X_SendEvent";
		case X_GrabPointer:
			return "X_GrabPointer";
		case X_UngrabPointer:
			return "X_UngrabPointer";
		case X_GrabButton:
			return "X_GrabButton";
		case X_UngrabButton:
			return "X_UngrabButton";
		case X_ChangeActivePointerGrab:
			return "X_ChangeActivePointerGrab";
		case X_GrabKeyboard:
			return "X_GrabKeyboard";
		case X_UngrabKeyboard:
			return "X_UngrabKeyboard";
		case X_GrabKey:
			return "X_GrabKey";
		case X_UngrabKey:
			return "X_UngrabKey";
		case X_AllowEvents:
			return "X_AllowEvents";
		case X_GrabServer:
			return "X_GrabServer";
		case X_UngrabServer:
			return "X_UngrabServer";
		case X_QueryPointer:
			return "X_QueryPointer";
		case X_GetMotionEvents:
			return "X_GetMotionEvents";
		case X_TranslateCoords:
			return "X_TranslateCoords";
		case X_WarpPointer:
			return "X_WarpPointer";
		case X_SetInputFocus:
			return "X_SetInputFocus";
		case X_GetInputFocus:
			return "X_GetInputFocus";
		case X_QueryKeymap:
			return "X_QueryKeymap";
		case X_OpenFont:
			return "X_OpenFont";
		case X_CloseFont:
			return "X_CloseFont";
		case X_QueryFont:
			return "X_QueryFont";
		case X_QueryTextExtents:
			return "X_QueryTextExtents";
		case X_ListFonts:
			return "X_ListFonts";
		case X_ListFontsWithInfo:
			return "X_ListFontsWithInfo";
		case X_SetFontPath:
			return "X_SetFontPath";
		case X_GetFontPath:
			return "X_GetFontPath";
		case X_CreatePixmap:
			return "X_CreatePixmap";
		case X_FreePixmap:
			return "X_FreePixmap";
		case X_CreateGC:
			return "X_CreateGC";
		case X_ChangeGC:
			return "X_ChangeGC";
		case X_CopyGC:
			return "X_CopyGC";
		case X_SetDashes:
			return "X_SetDashes";
		case X_SetClipRectangles:
			return "X_SetClipRectangles";
		case X_FreeGC:
			return "X_FreeGC";
		case X_ClearArea:
			return "X_ClearArea";
		case X_CopyArea:
			return "X_CopyArea";
		case X_CopyPlane:
			return "X_CopyPlane";
		case X_PolyPoint:
			return "X_PolyPoint";
		case X_PolyLine:
			return "X_PolyLine";
		case X_PolySegment:
			return "X_PolySegment";
		case X_PolyRectangle:
			return "X_PolyRectangle";
		case X_PolyArc:
			return "X_PolyArc";
		case X_FillPoly:
			return "X_FillPoly";
		case X_PolyFillRectangle:
			return "X_PolyFillRectangle";
		case X_PolyFillArc:
			return "X_PolyFillArc";
		case X_PutImage:
			return "X_PutImage";
		case X_GetImage:
			return "X_GetImage";
		case X_PolyText8:
			return "X_PolyText8";
		case X_PolyText16:
			return "X_PolyText16";
		case X_ImageText8:
			return "X_ImageText8";
		case X_ImageText16:
			return "X_ImageText16";
		case X_CreateColormap:
			return "X_CreateColormap";
		case X_FreeColormap:
			return "X_FreeColormap";
		case X_CopyColormapAndFree:
			return "X_CopyColormapAndFree";
		case X_InstallColormap:
			return "X_InstallColormap";
		case X_UninstallColormap:
			return "X_UninstallColormap";
		case X_ListInstalledColormaps:
			return "X_ListInstalledColormaps";
		case X_AllocColor:
			return "X_AllocColor";
		case X_AllocNamedColor:
			return "X_AllocNamedColor";
		case X_AllocColorCells:
			return "X_AllocColorCells";
		case X_AllocColorPlanes:
			return "X_AllocColorPlanes";
		case X_FreeColors:
			return "X_FreeColors";
		case X_StoreColors:
			return "X_StoreColors";
		case X_StoreNamedColor:
			return "X_StoreNamedColor";
		case X_QueryColors:
			return "X_QueryColors";
		case X_LookupColor:
			return "X_LookupColor";
		case X_CreateCursor:
			return "X_CreateCursor";
		case X_CreateGlyphCursor:
			return "X_CreateGlyphCursor";
		case X_FreeCursor:
			return "X_FreeCursor";
		case X_RecolorCursor:
			return "X_RecolorCursor";
		case X_QueryBestSize:
			return "X_QueryBestSize";
		case X_QueryExtension:
			return "X_QueryExtension";
		case X_ListExtensions:
			return "X_ListExtensions";
		case X_ChangeKeyboardMapping:
			return "X_ChangeKeyboardMapping";
		case X_GetKeyboardMapping:
			return "X_GetKeyboardMapping";
		case X_ChangeKeyboardControl:
			return "X_ChangeKeyboardControl";
		case X_GetKeyboardControl:
			return "X_GetKeyboardControl";
		case X_Bell:
			return "X_Bell";
		case X_ChangePointerControl:
			return "X_ChangePointerControl";
		case X_GetPointerControl:
			return "X_GetPointerControl";
		case X_SetScreenSaver:
			return "X_SetScreenSaver";
		case X_GetScreenSaver:
			return "X_GetScreenSaver";
		case X_ChangeHosts:
			return "X_ChangeHosts";
		case X_ListHosts:
			return "X_ListHosts";
		case X_SetAccessControl:
			return "X_SetAccessControl";
		case X_SetCloseDownMode:
			return "X_SetCloseDownMode";
		case X_KillClient:
			return "X_KillClient";
		case X_RotateProperties:
		  	return "X_RotateProperties";
		case X_ForceScreenSaver:
			return "X_ForceScreenSaver";
		case X_SetPointerMapping:
			return "X_SetPointerMapping";
		case X_GetPointerMapping:
			return "X_GetPointerMapping";
		case X_SetModifierMapping:
		  	return "X_SetModifierMapping";
		case X_GetModifierMapping:
			return "X_GetModifierMapping";
		case X_NoOperation:
			return "X_NoOperation";
		default:
			return "Unknown Request";
    }
}

#if 0
char *ux11_error( XErrorEvent *evt)
/*
 * Returns a nicely formatted error message for the error `evt'.  The
 * message is returned in a static buffer that should not be modified.
 */
{
    char local_buf[MAX_ERR_LEN/2];
	char err_buf[MAX_ERR_LEN];

    XGetErrorText(evt->display, evt->error_code, local_buf, MAX_ERR_LEN/2);
    (void) sprintf(err_buf, "%s: %s\n (serial=%lu minor_code=%d resourceid=0x%x)\n",
		   req_str((int) evt->request_code), local_buf,
		   evt->serial,
		   (int) evt->minor_code,
		   (int) evt->resourceid
	);
    return (char *) strdup(err_buf);
}
#else

char *ux11_error( XErrorEvent *evt)
/*
 * Returns a nicely formatted error message for the error `evt'.  The
 * message is returned in a static buffer that should not be modified.
 */
{ char local_buf[MAX_ERR_LEN/2]= "";
  char err_buf[MAX_ERR_LEN], *c;

	sprintf( err_buf, "%d", evt->error_code );
    XGetErrorDatabaseText(evt->display, "XRequest", err_buf, "", local_buf, MAX_ERR_LEN/3);
	if( *local_buf ){
		strcat( local_buf, "\n " );
	}
	c= &local_buf[ strlen(local_buf) ];
    XGetErrorText(evt->display, evt->error_code, c, MAX_ERR_LEN/3);
    (void) sprintf(err_buf, "%s: %s\n (serial=%lu minor_code=%d resourceid=0x%x)\n",
		   req_str((int) evt->request_code), local_buf,
		   evt->serial,
		   (int) evt->minor_code,
		   (int) evt->resourceid
	);
    return (char *) strdup(err_buf);
}

#endif
