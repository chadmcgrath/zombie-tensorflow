const { app, BrowserWindow, powerSaveBlocker } = require('electron')
const path = require('path')
const isDev = !app.isPackaged

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      backgroundThrottling: false,
    }
  })

  if (isDev) {
    // In development, load from Vite dev server
    win.loadURL('http://localhost:3001')
    win.webContents.openDevTools()
  } else {
    // In production, load from built files
    win.loadFile(path.join(__dirname, 'dist/index.html'))
  }
}

app.whenReady().then(() => {
  const id = powerSaveBlocker.start('prevent-display-sleep')
  console.log('Power save blocker started:', powerSaveBlocker.isStarted(id))
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
