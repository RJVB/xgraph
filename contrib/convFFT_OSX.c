/* convolve_fft[&Data,&Mask,&Output,direction[,&Data_spectrum[,&Mask_spectrum]]]	*/
int ascanf_convolve_fft ( ASCB_ARGLIST )
{ ASCB_FRAME
  unsigned long i, direction= 1;
  ascanf_Function *Data= NULL, *Mask= NULL, *Output= NULL;
  ascanf_Function *Data_sp= NULL, *Mask_sp= NULL;
#ifdef DEBUG
  static int show_spec= 0;
#endif
#if defined(HAVE_FFTW) && HAVE_FFTW
  fftw_complex *mask= NULL, *mask_spec= NULL;
  fftw_plan mp, rp, ip;
#else
  char *mask= NULL, *mask_spec= NULL;
  typedef char fftw_complex;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0, NN, Nm= 0;
		if( !(Data= parse_ascanf_address(args[0], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data->iarray ){
			ascanf_emsg= " (invalid Data array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= Data->N;
			if( Data->malloc!= XG_fftw_malloc || Data->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= XG_fftw_malloc;
				ascanf_array_free= XG_fftw_free;
				Resize_ascanf_Array( Data, N, result );
			}
			NN= 2* (N/2+ 1);
		}
		if( (Mask= parse_ascanf_address(args[1], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) ){
			if( Mask->iarray ){
				ascanf_emsg= " (invalid Mask array argument (2)) ";
				ascanf_arg_error= 1;
			}
			else{
				Nm= Mask->N;
			}
		}
		if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Output->iarray ){
			ascanf_emsg= " (invalid Output array argument (3)) ";
			ascanf_arg_error= 1;
		}
		else if( Output->N!= NN || Output->malloc!= XG_fftw_malloc || Output->free!= XG_fftw_free ){
			Resize_ascanf_Array_force= True;
			ascanf_array_malloc= XG_fftw_malloc;
			ascanf_array_free= XG_fftw_free;
			Resize_ascanf_Array( Output, NN, result );
		}
		direction= (args[3])? 1 : 0;

		if( ascanf_arguments> 4 ){
			if( !(Data_sp= parse_ascanf_address(args[4], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data_sp->iarray ){
				ascanf_emsg= " (invalid data_spec array argument (5)) ";
				ascanf_arg_error= 1;
			}
			else if( Data_sp->N!= NN ){
				  /* No need to use special (de)allocators */
				Resize_ascanf_Array( Data_sp, NN, result );
			}
		}
		if( ascanf_arguments> 5 ){
			if( !(Mask_sp= parse_ascanf_address(args[5], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Mask_sp->iarray ){
				ascanf_emsg= " (invalid mask_spec array argument (6)) ";
				ascanf_arg_error= 1;
			}
			else if( Mask_sp->N!= NN ){
				if( !Mask ){
				  /* This means we re-use the mask-spectrum returned by a previous call
				   \ (or another initialisation). We must not touch it here, thus we
				   \ warn and return an error when the spectrum passed does not have the
				   \ right size.
				   */
					fprintf( StdErr, " (mask_spec array must have %d size to match Data[%d]) ",
						NN, N
					);
					ascanf_emsg= " (mask_spec array has wrong size) ";
					ascanf_arg_error= 1;
				}
				else{
					Resize_ascanf_Array( Mask_sp, NN, result );
				}
			}
		}
		if( !Mask && !Mask_sp ){
			ascanf_emsg= " (Must specify either valid Mask or mask_spec!) ";
			ascanf_arg_error= 1;
		}

		if( !ascanf_arg_error ){
			if( FFTW_Initialised ){
				mask= (fftw_complex*) XG_fftw_malloc( (NN/2)* sizeof(fftw_complex) );
				mask_spec= (fftw_complex*) XG_fftw_malloc( (NN/2)* sizeof(fftw_complex) );
			}
			else{
				mask= (fftw_complex*) malloc( (NN/2)* sizeof(fftw_complex) );
				mask_spec= (fftw_complex*) malloc( (NN/2)* sizeof(fftw_complex) );
			}
			if( !mask || !mask_spec ){
				fprintf( StdErr, " (can't get mask memory (%s)) ", serror() );
				ascanf_arg_error= 1;
			}
			else{
				memset( mask, 0, (NN/2)* sizeof(fftw_complex) );
				memset( mask_spec, 0, (NN/2)* sizeof(fftw_complex) );
			}
		}

		if( ascanf_arg_error || !N || (Mask && !Nm) || !Output->array || !mask || !mask_spec || ascanf_SyntaxCheck ){
			if( FFTW_Initialised ){
				fftw_xfree( mask );
				fftw_xfree( mask_spec );
			}
			else{
				xfree( mask );
				xfree( mask_spec );
			}
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW

		{ ALLOCA( data, double, N, data_len);
		  ALLOCA( output, double, Output->N, output_len);
		  int plevel;
			  /* FFTW3's planners want pointers to the to-be-used memory, and *overwrite* it. */
			memcpy( data, Data->array, N* sizeof(double));
			memcpy( output, Output->array, Output->N* sizeof(double) );

			CLIP( *fftw_planner_level, -1, 2 );
			switch( (int) *fftw_planner_level ){
				case -1:
					plevel= FFTW_ESTIMATE;
					break;
				case 0:
					plevel= FFTW_MEASURE;
					break;
				case 1:
					plevel= FFTW_PATIENT;
					break;
				case 2:
					plevel= FFTW_EXHAUSTIVE;
					break;
			}
			if( FFTW_Initialised ){
				if( Mask ){
					mp= fftw_plan_dft_r2c_1d( N, (double*) mask, mask_spec, plevel );
				}
				rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) Output->array, plevel );
				ip= fftw_plan_dft_c2r_1d( N, mask, Output->array, plevel );
			}
			else{
#ifdef FFTW_DYNAMIC
				ascanf_emsg= " (fftw not initialised) ";
				ascanf_arg_error= 1;
				xfree( mask );
				xfree( mask_spec );
				return(0);
#else
				if( Mask ){
					mp= fftw_plan_dft_r2c_1d( N, (double*) mask, mask_spec, FFTW_ESTIMATE );
				}
				rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) Output->array, FFTW_ESTIMATE );
				ip= fftw_plan_dft_r2c_1d( N, mask, Output->array, FFTW_ESTIMATE );
#endif
			}
			memcpy( Data->array, data, N* sizeof(double) );
			memcpy( Output->array, output, Output->N* sizeof(double) );
		}

		if( Mask ){
		  long j;
		  double *m= (double*) mask;
			  /* Put the real mask in the centre of the array to be FFTed
			   \ -- disregarding any padding at the end of that array!
			   */
			j= Nm-1;
			for( i= 0; i< N/2- Nm/2; i++ ){
				m[i]= Mask->array[j];
			}
			  /* 990914: put in the mask reversed	*/
			for( j= Nm-1; j>= 0; i++, j-- ){
				m[i]= Mask->array[j];
			}
			j= 0;
			for( ; i< N; i++ ){
				m[i]= Mask->array[j];
			}
			fftw_execute( mp );
		}
		else{
			memcpy( mask_spec, Mask_sp->array, (NN/2)* sizeof(fftw_complex) );
		}

		fftw_execute( rp );

		if( Data_sp && Data_sp->array ){
			memcpy( Data_sp->array, Output->array, (NN)* sizeof(double) );
			Data_sp->value= Data_sp->array[ (Data_sp->last_index= 0) ];
			if( Data_sp->accessHandler ){
				AccessHandler( Data_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}
		if( Mask && Mask_sp && Mask_sp->array ){
			memcpy( Mask_sp->array, mask_spec, (NN)* sizeof(double) );
			Mask_sp->value= Mask_sp->array[ (Mask_sp->last_index= 0) ];
			if( Mask_sp->accessHandler ){
				AccessHandler( Data_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}

		{ fftw_complex *dat= (fftw_complex*) Output->array;
		  int sign= 1;
			  /* multiply odd elements with -1 to get "the centre in the centre"
			   \ (and not at the 2 edges)
			   */
			if( direction ){
				for( i= 0; i<= N/2; i++ ){
					c_re(mask[i])= sign* (c_re(dat[i])* c_re(mask_spec[i])- c_im(dat[i])* c_im(mask_spec[i]))/ N;
					c_im(mask[i])= sign* (c_re(dat[i])* c_im(mask_spec[i])+ c_im(dat[i])* c_re(mask_spec[i]))/ N;
					sign*= -1;
				}
			}
			else{
			  double b2;
				  /* vi subst pattern for c_re()/c_im(): s/\([^ ^I\[(]*\)\[i\]\.\([ri][em]\)/c_\2(\1[i])/gc */
				for( i= 0; i<= N/2; i++ ){
					b2= N*( c_re(mask_spec[i])* c_re(mask_spec[i]) + c_im(mask_spec[i])* c_im(mask_spec[i]) );
					if( b2 ){
						c_re(mask[i])= sign* (c_re(dat[i])* c_re(mask_spec[i])+ c_im(dat[i])* c_im(mask_spec[i]))/ b2;
						c_im(mask[i])= sign* (-c_re(dat[i])* c_im(mask_spec[i])+ c_im(dat[i])* c_re(mask_spec[i]))/ b2;
					}
					else{
					  /* Define deconvolution with NULL mask as NOOP:	*/
						  /* fftw_complex can be a structure, or it can be a double[2], in which
						   \ case a direct assignment won't work. memmove() will handle all possible
						   \ cases (including overlapping memory).
						   */
						memmove( &mask[i], &dat[i], sizeof(fftw_complex) );
					}
					sign*= -1;
				}
			}
		}
#	ifdef DEBUG
		memcpy( Output->array, mask, NN* sizeof(double) );
		if( !show_spec ){
			fftw_execute( ip );
		}
#	else
		fftw_execute( ip );
#	endif
		fftw_destroy_plan(ip);
		fftw_destroy_plan(rp);
		fftw_destroy_plan(mp);
#endif
		if( FFTW_Initialised ){
			fftw_xfree( mask );
			fftw_xfree( mask_spec );
		}
		else{
			xfree( mask );
			xfree( mask_spec );
		}

		GCA();

		Output->value= Output->array[ (Output->last_index= 0) ];
		if( Output->accessHandler ){
			AccessHandler( Output, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		*result= N;
		return(1);
	}
	return(0);
}

