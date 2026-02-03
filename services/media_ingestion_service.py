import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import hashlib

try:
    import yt_dlp as youtube_dl
    HAS_YOUTUBE_DL = True
except ImportError:
    HAS_YOUTUBE_DL = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class MediaSource:
    """Represents a media source (video, URL, etc.)"""
    id: str
    source_type: str
    original_url: str
    title: str
    description: str
    upload_time: datetime
    user_id: str
    extracted_content: str = ""
    transcript: str = ""
    metadata: Dict = field(default_factory=dict)
    processing_status: str = "pending"
    error_message: str = ""
    file_path: Optional[str] = None
    duration: int = 0
    size: int = 0


class MediaIngestionService:
    """Service for handling video and URL uploads for training"""
    
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024
    MAX_URL_SIZE = 500 * 1024 * 1024 
    ALLOWED_VIDEO_FORMATS = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    SUPPORTED_PLATFORMS = ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com', 'ted.com']
    
    def __init__(self, media_dir: str = "media", cache_dir: str = "media_cache"):
        self.media_dir = Path(media_dir)
        self.cache_dir = Path(cache_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.media_db: Dict[str, MediaSource] = {}
        self.user_media: Dict[str, List[MediaSource]] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
    
    def _generate_media_id(self, content: str, source_type: str) -> str:
        """Generate unique ID for media source"""
        hash_content = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{source_type}_{timestamp}_{hash_content}"
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is YouTube"""
        return any(domain in url for domain in ['youtube.com', 'youtu.be', 'youtube-nocookie.com'])
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def process_url(self, url: str, user_id: str) -> MediaSource:
        """Process and download content from a URL"""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        media_id = self._generate_media_id(url, "url")
        
        media_source = MediaSource(
            id=media_id,
            source_type="url",
            original_url=url,
            title=url[:100],
            description="",
            upload_time=datetime.utcnow(),
            user_id=user_id,
            processing_status="processing"
        )
        
        try:
            if self._is_youtube_url(url):
                media_source = await self._process_youtube(url, user_id)
            else:
                media_source = await self._process_generic_url(url, user_id)
            
            media_source.processing_status = "completed"
            
        except Exception as e:
            media_source.processing_status = "failed"
            media_source.error_message = str(e)
        
        self.media_db[media_id] = media_source
        if user_id not in self.user_media:
            self.user_media[user_id] = []
        self.user_media[user_id].append(media_source)
        
        return media_source
    
    async def _process_youtube(self, url: str, user_id: str) -> MediaSource:
        """Process YouTube video"""
        if not HAS_YOUTUBE_DL:
            raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")
        
        media_id = self._generate_media_id(url, "youtube")
        
        try:
           
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': False,
            }
            
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            duration = info.get('duration', 0)
            
            transcript = await self._extract_youtube_transcript(url)
            
            extracted_content = f"Title: {title}\n\n"
            if description:
                extracted_content += f"Description:\n{description}\n\n"
            if transcript:
                extracted_content += f"Transcript:\n{transcript}"
            
            media_source = MediaSource(
                id=media_id,
                source_type="youtube",
                original_url=url,
                title=title,
                description=description,
                upload_time=datetime.utcnow(),
                user_id=user_id,
                extracted_content=extracted_content,
                transcript=transcript,
                duration=duration,
                processing_status="completed",
                metadata={
                    'video_id': self._extract_youtube_id(url),
                    'duration': duration,
                    'platform': 'YouTube'
                }
            )
            
            return media_source
            
        except Exception as e:
            raise Exception(f"Failed to process YouTube video: {str(e)}")
    
    async def _extract_youtube_transcript(self, url: str) -> str:
        """Extract transcript from YouTube video"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            video_id = self._extract_youtube_id(url)
            if not video_id:
                return ""
            
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([item['text'] for item in transcript_list])
                return transcript_text
            except Exception:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=['en', 'es', 'fr']
                    )
                    transcript_text = " ".join([item['text'] for item in transcript_list])
                    return transcript_text
                except:
                    return ""
                    
        except ImportError:
            return ""
        except Exception:
            return ""
    
    async def _process_generic_url(self, url: str, user_id: str) -> MediaSource:
        """Process generic URL and extract text content"""
        media_id = self._generate_media_id(url, "url")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        raise Exception(f"URL returned status {response.status}")
                    
                    content_type = response.headers.get('content-type', '')
                    content = await response.read()
                    
                    if len(content) > self.MAX_URL_SIZE:
                        raise ValueError(f"Content size exceeds maximum of {self.MAX_URL_SIZE // (1024*1024)}MB")
                    
                    path = urlparse(url).path
                    title = path.split('/')[-1] or url.split('//')[1].split('/')[0]
                    
                    extracted_content = await self._extract_url_content(content, content_type, url)
                    
                    media_source = MediaSource(
                        id=media_id,
                        source_type="url",
                        original_url=url,
                        title=title,
                        description=f"Content from {urlparse(url).netloc}",
                        upload_time=datetime.utcnow(),
                        user_id=user_id,
                        extracted_content=extracted_content,
                        size=len(content),
                        processing_status="completed",
                        metadata={
                            'content_type': content_type,
                            'domain': urlparse(url).netloc
                        }
                    )
                    
                    return media_source
                    
        except Exception as e:
            raise Exception(f"Failed to process URL: {str(e)}")
    
    async def _extract_url_content(self, content: bytes, content_type: str, url: str) -> str:
        """Extract text content from URL"""
        extracted = f"Source: {url}\n\n"
        
        try:
            if 'text/html' in content_type or 'text/plain' in content_type:
                if 'text/html' in content_type:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        title_tag = soup.find('h1') or soup.find('title')
                        if title_tag:
                            extracted += f"Title: {title_tag.get_text()}\n\n"
                        
                        main_content = soup.find('main') or soup.find('article') or soup.find('body')
                        if main_content:
                            text = main_content.get_text(separator="\n", strip=True)
                            extracted += text[:5000]  
                        
                    except ImportError:
                        try:
                            text = content.decode('utf-8')
                            extracted += text[:5000]
                        except:
                            extracted += "[Content could not be extracted]"
                else:
                    try:
                        text = content.decode('utf-8')
                        extracted += text[:5000]
                    except:
                        extracted += "[Content could not be decoded]"
            
            elif 'application/json' in content_type:
                try:
                    data = json.loads(content.decode('utf-8'))
                    extracted += json.dumps(data, indent=2)[:5000]
                except:
                    extracted += "[JSON could not be parsed]"
            
            elif 'application/pdf' in content_type:
                extracted += "[PDF detected - please use file upload for PDF processing]"
            
            else:
                extracted += f"[Content-Type: {content_type}]"
        
        except Exception as e:
            extracted += f"[Error extracting content: {str(e)}]"
        
        return extracted
    
    async def process_video_file(self, file_content: bytes, filename: str, user_id: str) -> MediaSource:
        """Process uploaded video file"""
        if len(file_content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum of {self.MAX_FILE_SIZE // (1024*1024)}MB")
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.ALLOWED_VIDEO_FORMATS:
            raise ValueError(f"Unsupported video format: {file_ext}")
        
        media_id = self._generate_media_id(file_content[:1000] + filename.encode(), "video")
        
        file_path = self.media_dir / f"{media_id}{file_ext}"
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        media_source = MediaSource(
            id=media_id,
            source_type="video_file",
            original_url=filename,
            title=filename,
            description=f"Uploaded video file",
            upload_time=datetime.utcnow(),
            user_id=user_id,
            file_path=str(file_path),
            size=len(file_content),
            processing_status="processing"
        )
        
        try:
            duration, fps, resolution = await self._extract_video_metadata(file_path)
            media_source.duration = duration
            media_source.metadata = {
                'fps': fps,
                'resolution': resolution,
                'duration': duration,
                'filename': filename
            }
            
            transcript = await self._extract_video_audio_transcript(file_path)
            media_source.transcript = transcript
            
            extracted_content = f"Video: {filename}\n"
            extracted_content += f"Duration: {duration} seconds\n"
            extracted_content += f"Resolution: {resolution}\n\n"
            if transcript:
                extracted_content += f"Transcript:\n{transcript}"
            
            media_source.extracted_content = extracted_content
            media_source.processing_status = "completed"
            
        except Exception as e:
            media_source.processing_status = "completed"
            media_source.error_message = f"Partial processing: {str(e)}"
        
        self.media_db[media_id] = media_source
        if user_id not in self.user_media:
            self.user_media[user_id] = []
        self.user_media[user_id].append(media_source)
        
        return media_source
    
    async def _extract_video_metadata(self, file_path: Path) -> Tuple[int, float, str]:
        """Extract video metadata"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(file_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = int(frame_count / fps) if fps > 0 else 0
            resolution = f"{width}x{height}"
            
            cap.release()
            
            return duration, fps, resolution
            
        except ImportError:
            return 0, 0, "unknown"
        except Exception:
            return 0, 0, "unknown"
    
    async def _extract_video_audio_transcript(self, file_path: Path) -> str:
        """Extract audio from video and generate transcript"""
        try:
            import subprocess
            
            audio_path = file_path.with_suffix('.wav')
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-q:a', '9',
                '-n',
                str(audio_path)
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=300)
                
                if audio_path.exists():
                    transcript = await self._transcribe_audio(audio_path)
                    
                    try:
                        audio_path.unlink()
                    except:
                        pass
                    
                    return transcript
            except:
                pass
            
        except Exception:
            pass
        
        return ""
    
    async def _transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            with open(audio_path, 'rb') as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            
            return transcript.text
            
        except Exception:
            pass
        
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(str(audio_path)) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception:
            return ""
    
    def get_media_info(self, media_id: str) -> Optional[MediaSource]:
        """Get information about a media source"""
        return self.media_db.get(media_id)
    
    def get_user_media(self, user_id: str) -> List[MediaSource]:
        """Get all media sources for a user"""
        return self.user_media.get(user_id, [])
    
    def get_extracted_content(self, user_id: str, max_items: int = 5) -> List[str]:
        """Get extracted content from user's media for training"""
        user_media = self.get_user_media(user_id)
        content_list = []
        
        for media in sorted(user_media, key=lambda m: m.upload_time, reverse=True)[:max_items]:
            if media.extracted_content:
                content_list.append(media.extracted_content)
        
        return content_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        total_media = len(self.media_db)
        by_type = {}
        for media in self.media_db.values():
            by_type[media.source_type] = by_type.get(media.source_type, 0) + 1
        
        return {
            "total_media_sources": total_media,
            "by_type": by_type,
            "total_users": len(self.user_media),
            "initialized": True
        }

media_ingestion_service = MediaIngestionService()
