import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

interface GameGalleryProps {
    appIds: number[];
    maxSelections?: number;
  }
  
  const GameGallery: React.FC<GameGalleryProps> = ({ appIds, maxSelections = 5 }) => {
    const [gamePhotos, setGamePhotos] = useState<Record<number, string | null>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedGames, setSelectedGames] = useState<number[]>([]);
    const [selectionLimitReached, setSelectionLimitReached] = useState(false);
  
  
    const getGamePhoto = (appId: number): string => {
      return `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appId}/header.jpg`;
    };

  useEffect(() => {
    const fetchGamePhotos = async () => {
      try {
        const promises = appIds.map(async (appId) => {
          const imageUrl = getGamePhoto(appId);
           const response = await fetch(imageUrl);
          if (!response.ok) {
              console.warn(`Image not found for appID: ${appId}`);
              return { appId, imageUrl: null };
          }
          return { appId, imageUrl };
        });

        const results = await Promise.all(promises);
        const photos: Record<number, string | null> = {};
        results.forEach(({ appId, imageUrl }) => {
            photos[appId] = imageUrl;
        });
        setGamePhotos(photos);

      } catch (err) {
        console.error("Error fetching game photos:", err);
        setError("Error fetching game photos. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchGamePhotos();
  }, [appIds]);

  const handleGameClick = (appId: number) => {
    if (selectedGames.includes(appId)) {
      setSelectedGames(selectedGames.filter((id) => id !== appId));
      setSelectionLimitReached(false);
    } else if (selectedGames.length < maxSelections) {
      setSelectedGames([...selectedGames, appId]);
    } else {
        setSelectionLimitReached(true);
    }
  };

  const isSelected = (appId: number) => selectedGames.includes(appId);

  if (loading) {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {appIds.map((appId) => (
                <Card key={appId}>
                    <CardHeader>
                      <Skeleton className="h-48 w-full" />
                    </CardHeader>
                </Card>
            ))}
        </div>
    );
  }

  if (error) {
    return <div className="text-center text-red-500">{error}</div>;
  }

  return (
        <div>
            {selectionLimitReached && (
            <p className="text-red-500 mb-2">You can only select up to {maxSelections} games.</p>
            )}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {appIds.map((appId) => (
            <Card
                key={appId}
                className={`cursor-pointer ${isSelected(appId) ? 'border-4 border-blue-500' : ''}`}
                onClick={() => handleGameClick(appId)}
            >
                <CardHeader>
                    {gamePhotos[appId] ? (
                    <img
                        src={gamePhotos[appId]!}
                        alt={`Game ${appId} Header`}
                        className="w-full h-auto object-cover rounded-t-lg"
                        loading="lazy"
                    />
                    ) : (
                    <div className="bg-gray-200 h-48 flex items-center justify-center rounded-t-lg">
                        <p className="text-gray-500">No Image Available</p>
                    </div>
                    )}
                </CardHeader>
            </Card>
            ))}
        </div>
            <div className="mt-4">
                <p>Selected Games: {selectedGames.length} / {maxSelections}</p>
                {selectedGames.length > 0 && (
                    <p>Selected App IDs: {selectedGames.join(', ')}</p>
                )}
            </div>
        </div>
    );
};

export default GameGallery;