import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

describe('App Component', () => {
  it('renders the heading "Vite + React"', () => {
    render(<App />)
    const heading = screen.getByRole('heading', { name: /Vite \+ React/i })
    expect(heading).toBeInTheDocument()
  })

  it('increments count when the button is clicked', async () => {
    render(<App />)
    // Find the button with the initial text "count is 0"
    const button = screen.getByRole('button', { name: /count is 0/i })
    // Simulate a click event
    await userEvent.click(button)
    // Verify that the button now shows "count is 1"
    expect(button).toHaveTextContent('count is 1')
  })
})