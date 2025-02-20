namespace GameRecommender.Interfaces.Data;

public interface IGameGalleryHandler
{
    Task<bool> GameChosenInGallery(Guid userId);
}