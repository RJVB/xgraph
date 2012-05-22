	if( wi->FitOnce> 0 && wi->fit_xbounds> 0 && wi->fit_ybounds> 0 ){
		wi->FitOnce= 0;
	}
	if( wi->FitOnce> 0 ){
	  /* We'll fit the data to the window just once. Meaning we'll set
	   \ the fitting flags to true, (re)storing the original values. This
	   \ will not make any difference when they're already on. We use static variables
	   \ and a switch to indicate whether the values are already stored, to prevent (?)
	   \ a lasting effect (= turning on fit_xy scaling), which occurred sometimes (hitting
	   \ Mod1-s twice in succession?).
	   */
		if( !fit_xy_set ){
			fit_x= wi->fit_xbounds;
			fit_y= wi->fit_ybounds;
		}
		fit_xy_set= True;
		if( !wi->polarFlag ){
			wi->fit_xbounds= 1;
		}
		else{
			wi->fit_xbounds= 0;
		}
		wi->fit_ybounds= 1;
		wi->FitOnce= -1;
	}
	else if( wi->FitOnce== -1 ){
		wi->FitOnce-= 1;
	}
	if( wi->fit_xbounds> 0 || wi->fit_ybounds> 0 ){
	  int rd= wi->redraw;
	  int fx= wi->fit_xbounds;
	  int fy= wi->fit_ybounds;
	  int silent;
		  /* determine the current data's bounds by calling DrawData
		   \ with the bounds_only flag set.
		   */
		wi->redraw= 0;
		if( wi->FitOnce ){
			XStoreName(disp, wi->window, "Incidental rescaling..." );
		}
		else{
			XStoreName(disp, wi->window, "(Re)Scaling..." );
		}
		XFlush( disp);
		if( ( !wi->raw_display && wi->process_bounds && 
				(wi->transform.x_len || wi->process.data_process_len || wi->transform.y_len )
			) || wi->polarFlag
		){
			/* silent redraw handled by the Fit_.Bounds() functions.	*/
		}
		else{
			silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			DrawData( wi, True );
			if( wi->delete_it== -1 || wi->dev_info.user_state== NULL ){
				ActiveWin= NULL;
				if( wi->FitOnce== -1 ){
					if( fit_xy_set ){
						wi->fit_xbounds= fit_x;
						wi->fit_ybounds= fit_y;
					}
					fit_xy_set= False;
					wi->FitOnce= False;
				}
				ret_code= 0; goto DrawWindow_return;
			}
			wi->dev_info.xg_silent( wi->dev_info.user_state, silent );
		}
		if( wi->fit_xbounds || wi->fit_ybounds ){
			  /* Fit_XBounds() maybe calls us again, so we should prevent endless loops.
			   \ Also, when we are re-called to establish the X-bounds, we do not have to
			   \ do the Y-bounds then!
			   */
			wi->fit_xbounds*= -1;
			wi->fit_ybounds*= -1;
			if( wi->fit_xbounds && wi->fit_ybounds ){
				Fit_XYBounds( wi );
			}
			else if( wi->fit_xbounds ){
				Fit_XBounds( wi );
			}
			else if( wi->fit_ybounds ){
				Fit_YBounds( wi );
			}
			wi->fit_xbounds= fx;
			wi->fit_ybounds= fy;
		}
		wi->redraw= rd;
	}
	if( wi->FitOnce== -1 ){
/* 		wi->fit_xbounds= fit_x;	*/
/* 		wi->fit_ybounds= fit_y;	*/
		if( fit_xy_set ){
			wi->fit_xbounds= fit_x;
			wi->fit_ybounds= fit_y;
		}
		fit_xy_set= False;
		wi->FitOnce= False;
	}
	else{
		wi->FitOnce+= 1;
	}

