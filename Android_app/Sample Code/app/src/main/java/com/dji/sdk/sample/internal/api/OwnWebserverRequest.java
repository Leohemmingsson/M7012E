package com.dji.sdk.sample.internal.api;

import android.os.AsyncTask;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class OwnWebserverRequest extends AsyncTask<String, Void, String> {

    private static final String TAG = "HttpGetRequest";

    @Override
    protected String doInBackground(String... params) {
        String urlString = params[0];
        String result = null;
        HttpURLConnection urlConnection = null;

        try {
            URL url = new URL(urlString);
            urlConnection = (HttpURLConnection) url.openConnection();

            InputStream inputStream = urlConnection.getInputStream();
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            StringBuilder stringBuilder = new StringBuilder();
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }

            result = stringBuilder.toString();
        } catch (IOException e) {
            Log.e(TAG, "Error making GET request", e);
        } finally {
            if (urlConnection != null) {
                urlConnection.disconnect();
            }
        }

        return result;
    }

    @Override
    protected void onPostExecute(String result) {
        // Handle the result here, for example, update UI
        Log.d(TAG, "GET request result: " + result);
    }
}
