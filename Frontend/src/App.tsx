import viteLogo from '/vite.svg'
import { ThemeProvider } from '@/components/theme-provider'
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import { Button } from './components/ui/button'
import './App.css'

function App() {
  const [count, setCount] = useState(0);

  return (
    <ThemeProvider defaultTheme='dark' storageKey='vite-ui-theme'>
      <p className='text-red-300'>
        asdfasdfasdf
      </p>
      <Button>
        musisz tego sluchac glosno
      </Button>
    </ThemeProvider>
  );
}

export default App;
