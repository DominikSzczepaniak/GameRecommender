using GameRecommender.Interfaces.Data;

namespace GameRecommender.Data;

public class PostgresAppIdMappingHandler(PostgresConnectionPool connectionPool) : PostgresBase(connectionPool), IAppIdMappingHandler
{
    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO AppIdToNames (AppId, Name) VALUES (@AppId, @Name)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", appId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Name", name));

        await command.ExecuteNonQueryAsync();
    }
}