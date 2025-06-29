import requests

TMDB_API_KEY = "0f69e89b1f60756123aaaad6b5e448de"

def test_poster_fetch(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Request failed with status: {response.status_code}")
        return

    data = response.json()
    if not data['results']:
        print("❌ No movie found")
        return

    poster_path = data['results'][0].get('poster_path')
    if not poster_path:
        print("⚠️ Movie found, but no poster available")
        return

    poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    print(f"✅ Poster URL: {poster_url}")

# 🔽 Test with a movie title
test_poster_fetch("Toy Story")
