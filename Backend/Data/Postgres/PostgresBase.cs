using Npgsql;

namespace GameRecommender.Data;

public abstract class PostgresBase(PostgresConnectionPool connectionPool)
{
    public async Task<NpgsqlConnection> GetConnectionAsync()
    {
        return await connectionPool.GetConnectionAsync();
    }
}