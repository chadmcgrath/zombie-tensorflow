import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

describe('App Component', () => {
  it('renders the main game heading', () => {
    render(<App />)
    const heading = screen.getByRole('heading', { name: /AI Zombie Apocalypse Super Quest/i })
    expect(heading).toBeInTheDocument()
  })

  it('renders the subtitle heading', () => {
    render(<App />)
    const subtitle = screen.getByRole('heading', { name: /you just kinda watch it for a while/i })
    expect(subtitle).toBeInTheDocument()
  })

  it('renders game control buttons', () => {
    render(<App />)
    // Check for some key game control buttons
    const addHumanButton = screen.getByRole('button', { name: /Add Human/i })
    const addZombieButton = screen.getByRole('button', { name: /Add Zombie/i })
    const saveButton = screen.getByRole('button', { name: /Save/i })
    const loadButton = screen.getByRole('button', { name: /Load/i })
    
    expect(addHumanButton).toBeInTheDocument()
    expect(addZombieButton).toBeInTheDocument()
    expect(saveButton).toBeInTheDocument()
    expect(loadButton).toBeInTheDocument()
  })

  it('renders game speed control', () => {
    render(<App />)
    const gameSpeedSlider = screen.getByRole('slider', { name: /Game Speed/i })
    expect(gameSpeedSlider).toBeInTheDocument()
  })

  it('can interact with reward configuration button', async () => {
    render(<App />)
    const rewardConfigButton = screen.getByRole('button', { name: /Reward Configuration/i })
    expect(rewardConfigButton).toBeInTheDocument()
    
    // Test that we can click the button (this should open a modal)
    await userEvent.click(rewardConfigButton)
    // The modal should now be visible with reward inputs
    const hitShotRewardInput = screen.getByRole('spinbutton', { name: /Hit Shot Reward/i })
    expect(hitShotRewardInput).toBeInTheDocument()
  })
})
