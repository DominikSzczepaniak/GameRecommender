//initial: puste cardy + guzik rekomendacji
//po guziku (call api) -> wyswietlic k cardow

import { GameCard } from '@/components/GameCard';
import { GetRecommendation } from '@/components/GetRecommendation';
import { Button } from '@/components/ui/button';
import { GameData } from '@/models/Game';
import { API_SERVER } from '@/settings';

const Home = () => {
  const handleGenerate = async (): Promise<GameData[]> => {
    const response = await fetch(`${API_SERVER}/recommendations/1`, {
      method: 'POST',
      headers: {
        'Content-Type': 'applications/json',
        'Authentication': `Bearer ${localStorage.getItem('token')}`,
      },
      body: JSON.stringify({
        user: {
          Id: localStorage.getItem('userId'),
          Username: localStorage.getItem('username'),
          Email: localStorage.getItem('email'),
          Password: '',
        },
      }),
    });
    if (!response.ok) {
      throw new Error('Request failed');
    }
    const data: GameData[] = await response.json();
    return data;
  };

  const handleLike = async (appId: string): Promise<void> => {
  };

  const handleDislike = async (appId: string): Promise<void> => {
  };

  return (
    <div>
      <Button onClick={handleGenerate}>
        Generate
      </Button>
      <GetRecommendation
        recommendationEngines={[{
          name: 'lightFM',
          likeFunction: handleLike,
          dislikeFunction: handleDislike,
          askForRecommendations: handleGenerate,
        }]} />
    </div>
  );
};

export default Home;
