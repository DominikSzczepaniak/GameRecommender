using GameRecommender.Interfaces.Data;
using Npgsql;

namespace GameRecommender.Data;

public class PostgresGameGalleryHandler(PostgresConnectionPool connectionPool) : PostgresBase(connectionPool), IGameGalleryHandler
{
    public async Task<bool> GameChosenInGallery(Guid userId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM UserGaleryChosen WHERE UserId = @UserId";
        command.Parameters.Add(new NpgsqlParameter("@UserId", userId));

        using var reader = await command.ExecuteReaderAsync();
        if (await reader.ReadAsync())
        {
            return reader.GetBoolean(reader.GetOrdinal("Chosen"));
        }

        return false;
    }
}