/*
 * X11 Utility Functions
 */

char *ux11_get_value(argc, argv, value, def)
int argc;
char *argv[];
char *value;
char *def;
/*
 * Reads through all arguments looking for `value' (which is
 * usually a name preceeded by a - sign) and returns the
 * value found.  If no value is found,  def is returned.
 */
{
    int idx;
    char *ret_value = def;

    for (idx = 1;  idx < argc-1;  idx++) {
	if (strcmp(argv[idx], value) == 0) {
	    ret_value = argv[idx+1];
	    break;
	}
    }
    return ret_value;
}

