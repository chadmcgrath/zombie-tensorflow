const { app, BrowserWindow, powerSaveBlocker } = require('electron')
app.whenReady()
function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      backgroundThrottling: false, 
    }
  })

  win.loadFile('indexShips.html') 
}

app.whenReady().then(() => {
  const id = powerSaveBlocker.start('prevent-display-sleep')
  console.log(powerSaveBlocker.isStarted(id))
  createWindow();
});