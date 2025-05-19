import './App.css'
import { ContainerQServer, Tiled, HubAppLayout, RouteItem } from 'bluesky-web'
import 'bluesky-web/style.css';
import ControlPage from './pages/ControlPage';
import { House, Joystick, StackPlus, ImageSquare  } from "@phosphor-icons/react";

function App() {
  const routes:RouteItem[] = [
    {element:<div>About</div>, path: "/", label: "Home", icon: <House size={32} />},
    {element:<ControlPage/>, path: "/control", label: "Control", icon: <Joystick size={32} />},
    {element:<ContainerQServer className="m-8 h-[calc(100%-4rem)] w-[calc(100%-4rem)] bg-white/50"/>, path: "/qserver", label: "Q Server", icon: <StackPlus size={32} />},
    {element:<Tiled />, path: "/data", label: "Data", icon: <ImageSquare size={32} />},
  ]
  return (
    <HubAppLayout routes={routes}/>
  )
}

export default App
