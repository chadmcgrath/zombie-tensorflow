const { app, BrowserWindow, powerSaveBlocker } = require('electron')
app.whenReady()
function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      backgroundThrottling: false, // Add this line
    }
  })

  win.loadFile('src/index.html')  // Load your HTML file
}

app.whenReady().then(() => {
  const id = powerSaveBlocker.start('prevent-app-suspension')
  console.log(powerSaveBlocker.isStarted(id))
  createWindow();
});