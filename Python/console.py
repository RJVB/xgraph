import wx
import code
import sys
import traceback

class ConsoleRedirect:
    '''used to redirect stderr/stdout to a console'''
    def __init__(self, console):
        self.console = console
    def write(self, s):
        self.console.PrintString(s)


class wxConsole(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title)
        self.redir = ConsoleRedirect(self)
        sys.stderr = self.redir
        sys.stdout = self.redir

        self.source = ''
        self.prompt = '>>> '

        self.output = wx.TextCtrl(self, 1, style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.input = wx.TextCtrl(self, 1, style=wx.TE_PROCESS_ENTER)

        # fixed width fonts are best
        font = wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL, False, u'Courier New')
        self.output.SetFont(font)
        self.input.SetFont(font)

        # layout
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.output, 1, wx.EXPAND)
        self.vbox.Add(self.input, 0, wx.EXPAND)
        self.SetSizer(self.vbox)
        self.SetAutoLayout(1)
        self.Show()

        # events
        self.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        self.input.Bind(wx.EVT_KEY_DOWN, self.OnChar)
        self.input.WriteText(self.prompt)

        # misc
        self.input.SetFocus()
        self.input.SetSelection(4,4)


    def OnChar(self, event):
        # escape clears input line
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.input.Clear()
            self.input.WriteText(self.prompt)
            return

        # can't overwrite the prompt
        if self.input.GetInsertionPoint() == 4:
            if event.GetKeyCode() == wx.WXK_LEFT:
                return
            if event.GetKeyCode() == wx.WXK_BACK:
                return

        event.Skip()


    def OnEnter(self, event):
        cmd = self.input.GetValue()
        self.PrintString(cmd+'\n')
        self.input.Clear()
        self.ExecString(cmd)


    def ExecString(self, cmd):
        cmd = cmd[4:]
        self.source += '\n'+cmd
        c = code.compile_command(self.source)

        if c != None:
            # python code runs in the __main__ namespace
            # it might be useful to change this
            import __main__
            try:
                exec c in __main__.__dict__
            except:
                traceback.print_exc()
            self.source = ''
            self.input.Clear()
            self.prompt = '>>> '
        else:
            self.input.Clear()
            self.prompt = '... '

        self.input.WriteText(self.prompt)


    def PrintString(self, s):
        self.output.WriteText(s)

def wxConsoleStart():
    app = wx.App()
    console = wxConsole(None, -1, 'Console')
    app.MainLoop()

if __name__ == '__main__':
    wxConsoleStart()
