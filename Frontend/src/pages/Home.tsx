import { GetRecommendation } from '@/components/GetRecommendation';
import { GameData } from '@/models/Game';
import { API_SERVER } from '@/settings';

const Home = () => {
  const askForRecommendations = async (): Promise<GameData[]> => {
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

  const sendOpinion = async (appId: string, opinion: boolean) => {
    const response = await fetch(`${API_SERVER}/Game/addOpinion`, {
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
        gameDto: {
          AppId: appId,
          Opinion: opinion,
        },
      }),
    });
    if (!response.ok) {
      throw new Error('Request failed');
    }
  };

  const handleLike = async (appId: string): Promise<void> => {
    await sendOpinion(appId, true);
  };

  const handleDislike = async (appId: string): Promise<void> => {
    await sendOpinion(appId, false);
  };

  return (
    <div>
      <GetRecommendation
        recommendationEngines={[{
          name: 'lightFM',
          likeFunction: handleLike,
          dislikeFunction: handleDislike,
          askForRecommendations: askForRecommendations,
        }]} />
    </div>
  );
};

export default Home;
